// jarvis-aec: an echo-cancelled microphone and speaker for the voice line.
//
// Apple's voice processing I/O unit (the FaceTime pipeline) plays what the
// voice line sends it and records the mic with that playback removed, so the
// mic hears the person and not the voice coming back out of the speakers.
// Measured on this MacBook Air (2026-09-14): a spoken clip that reached the mic
// at -30 dBFS through the plain path came back at the -64 dBFS noise floor, and
// Whisper transcribed nothing from it.
//
// Findings that shaped this (see the Voice Line note in the vault):
//   - AVAudioEngine cannot be pointed at a separate mic and speaker with voice
//     processing on (-10875 / -10851). The raw VoiceProcessingIO unit can: the
//     mic is set on element 1 and the speaker on element 0.
//   - Voice processing ducks every other sound on the Mac by default, like a
//     call. Other-audio ducking is set to its minimum so YouTube and music keep
//     their volume.
//
// usage: jarvis-aec <mic device name> <speaker device name>
// stdin : frames of [1 byte kind][UInt32 little-endian length][payload]
//           'A'  int16 LE mono 24 kHz audio to play
//           'F'  flush: drop everything not yet played (barge-in)
//           'P'  pause: play silence, keep the queue (someone may be cutting in)
//           'R'  resume where it paused
//           'Q'  quit
//         EOF also quits, so the helper never outlives the voice line.
// stdout: the echo-cancelled mic, int16 LE mono 24 kHz, continuous
//         (the unit refuses a 16 kHz mic format: initialize -10875, checked
//         2026-09-14; backtalk/aec.py converts to the 16 kHz Whisper wants)
// stderr: "ready ..." once running, then log lines
import AudioToolbox
import CoreAudio
import Foundation

setvbuf(stderr, nil, _IONBF, 0)
signal(SIGPIPE, SIG_IGN)

let PLAY_RATE = 24000.0
let MIC_RATE = 24000.0

func log(_ s: String) {
    FileHandle.standardError.write(("[aec] " + s + "\n").data(using: .utf8)!)
}

// ---- devices ---------------------------------------------------------------

func allDevices() -> [AudioDeviceID] {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size)
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids)
    return ids
}

func deviceName(_ id: AudioDeviceID) -> String {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName,
                                          mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var cf: Unmanaged<CFString>? = nil
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &cf) == noErr, let s = cf else { return "" }
    return s.takeRetainedValue() as String
}

func channelCount(_ id: AudioDeviceID, input: Bool) -> Int {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration,
                                          mScope: input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput,
                                          mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, raw) == noErr else { return 0 }
    return UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        .reduce(0) { $0 + Int($1.mNumberChannels) }
}

func findDevice(_ want: String, input: Bool) -> AudioDeviceID? {
    let ids = allDevices()
    let usable = ids.filter { channelCount($0, input: input) > 0 }
    if let exact = usable.first(where: { deviceName($0) == want }) { return exact }
    return usable.first { deviceName($0).lowercased().contains(want.lowercased()) }
}

// ---- the speaker's queue (what is still to be played) ----------------------

final class PlayQueue {
    private var buf: [Float]
    private var head = 0
    private var count = 0
    private let lock = NSLock()

    init(seconds: Double) { buf = [Float](repeating: 0, count: Int(PLAY_RATE * seconds)) }

    func push(_ samples: [Float]) {
        lock.lock()
        for s in samples {
            if count == buf.count { head = (head + 1) % buf.count; count -= 1 }   // full: drop the oldest
            buf[(head + count) % buf.count] = s
            count += 1
        }
        lock.unlock()
    }

    // paused: the speaker plays silence and the queue keeps its place, so
    // resume picks up mid-word (a false barge-in costs a beat, not the reply)
    var paused = false

    func pop(_ out: UnsafeMutablePointer<Float>, _ n: Int) {
        lock.lock()
        let k = paused ? 0 : min(n, count)
        for i in 0..<k { out[i] = buf[(head + i) % buf.count] }
        head = (head + k) % buf.count
        count -= k
        lock.unlock()
        for i in k..<n { out[i] = 0 }
    }

    func clear() {
        lock.lock(); head = 0; count = 0; paused = false; lock.unlock()
    }

    func setPaused(_ p: Bool) {
        lock.lock(); paused = p; lock.unlock()
    }

    var queuedSeconds: Double {
        lock.lock(); defer { lock.unlock() }
        return Double(count) / PLAY_RATE
    }
}

// ---- the mic's outbox (what is still to be written to stdout) --------------

final class MicOutbox {
    private var pending = Data()
    private let lock = NSLock()
    private let ready = DispatchSemaphore(value: 0)

    func add(_ p: UnsafePointer<Float>, _ n: Int) {
        var chunk = Data(count: n * 2)
        chunk.withUnsafeMutableBytes { raw in
            let out = raw.bindMemory(to: Int16.self)
            for i in 0..<n { out[i] = Int16(max(-1, min(1, p[i])) * 32767).littleEndian }
        }
        lock.lock(); pending.append(chunk); lock.unlock()
        ready.signal()
    }

    func writer() {
        while true {
            ready.wait()
            lock.lock(); let data = pending; pending = Data(); lock.unlock()
            if data.isEmpty { continue }
            let ok = data.withUnsafeBytes { raw -> Bool in
                var off = 0
                while off < raw.count {
                    let w = write(1, raw.baseAddress! + off, raw.count - off)
                    if w <= 0 { return false }
                    off += w
                }
                return true
            }
            if !ok { log("stdout closed, exiting"); exit(0) }
        }
    }
}

// ---- globals the C audio callbacks reach (they cannot capture) --------------

let playQueue = PlayQueue(seconds: 120)
let micOutbox = MicOutbox()
var unit: AudioUnit? = nil
let micScratch = UnsafeMutablePointer<Float>.allocate(capacity: 8192)

let renderCB: AURenderCallback = { _, _, _, _, frames, ioData in
    guard let io = ioData else { return noErr }
    for buf in UnsafeMutableAudioBufferListPointer(io) {
        guard let m = buf.mData else { continue }
        playQueue.pop(m.assumingMemoryBound(to: Float.self), Int(buf.mDataByteSize) / 4)
    }
    return noErr
}

let inputCB: AURenderCallback = { _, flags, ts, _, frames, _ in
    let n = min(Int(frames), 8192)
    var abl = AudioBufferList(mNumberBuffers: 1,
                              mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(n * 4), mData: micScratch))
    let s = AudioUnitRender(unit!, flags, ts, 1, UInt32(n), &abl)
    if s != noErr { return s }
    micOutbox.add(micScratch, n)
    return noErr
}

// ---- setup -----------------------------------------------------------------

func check(_ what: String, _ status: OSStatus) {
    if status != noErr {
        log("\(what) failed: \(status)")
        exit(1)
    }
}

func setProp<T>(_ what: String, _ id: AudioUnitPropertyID, _ scope: AudioUnitScope, _ element: AudioUnitElement, _ value: T) {
    var v = value
    check(what, AudioUnitSetProperty(unit!, id, scope, element, &v, UInt32(MemoryLayout<T>.size)))
}

let args = CommandLine.arguments
guard args.count == 3 else {
    log("usage: jarvis-aec <mic device name> <speaker device name>")
    exit(2)
}
guard let micID = findDevice(args[1], input: true) else { log("mic \"\(args[1])\" not found"); exit(3) }
guard let spkID = findDevice(args[2], input: false) else { log("speaker \"\(args[2])\" not found"); exit(3) }

var desc = AudioComponentDescription(componentType: kAudioUnitType_Output,
                                     componentSubType: kAudioUnitSubType_VoiceProcessingIO,
                                     componentManufacturer: kAudioUnitManufacturer_Apple,
                                     componentFlags: 0, componentFlagsMask: 0)
guard let comp = AudioComponentFindNext(nil, &desc) else { log("no voice processing unit"); exit(1) }
check("new unit", AudioComponentInstanceNew(comp, &unit))

setProp("ducking", AudioUnitPropertyID(kAUVoiceIOProperty_OtherAudioDuckingConfiguration), kAudioUnitScope_Global, 0,
        AUVoiceIOOtherAudioDuckingConfiguration(mEnableAdvancedDucking: DarwinBoolean(false),
                                                mDuckingLevel: AUVoiceIOOtherAudioDuckingLevel(rawValue: 10)!))
setProp("enable mic", kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, UInt32(1))
setProp("enable speaker", kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, UInt32(1))
setProp("mic device", kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 1, micID)
setProp("speaker device", kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, spkID)

func mono(_ rate: Double) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription(mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
                                mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                                mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
                                mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
}
setProp("play format", kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, mono(PLAY_RATE))
setProp("mic format", kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, mono(MIC_RATE))
setProp("render callback", kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
        AURenderCallbackStruct(inputProc: renderCB, inputProcRefCon: nil))
setProp("input callback", kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0,
        AURenderCallbackStruct(inputProc: inputCB, inputProcRefCon: nil))
check("initialize", AudioUnitInitialize(unit!))
check("start", AudioOutputUnitStart(unit!))

Thread.detachNewThread { micOutbox.writer() }

func shutdown(_ code: Int32) -> Never {
    AudioOutputUnitStop(unit!)
    AudioUnitUninitialize(unit!)
    AudioComponentInstanceDispose(unit!)
    exit(code)
}

// read exactly n bytes from stdin; nil at EOF
func readExactly(_ n: Int) -> Data? {
    var data = Data(count: n)
    var got = 0
    let ok = data.withUnsafeMutableBytes { raw -> Bool in
        while got < n {
            let r = read(0, raw.baseAddress! + got, n - got)
            if r <= 0 { return false }
            got += r
        }
        return true
    }
    return ok ? data : (n == 0 ? Data() : nil)
}

log("ready mic=\"\(deviceName(micID))\" speaker=\"\(deviceName(spkID))\" play=\(Int(PLAY_RATE)) mic=\(Int(MIC_RATE))")

while true {
    guard let header = readExactly(5) else { shutdown(0) }
    let kind = header[0]
    let length = Int(header.subdata(in: 1..<5).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
    guard let payload = readExactly(length) else { shutdown(0) }
    switch kind {
    case UInt8(ascii: "A"):
        let samples: [Float] = payload.withUnsafeBytes { raw in
            raw.bindMemory(to: Int16.self).map { Float(Int16(littleEndian: $0)) / 32768.0 }
        }
        playQueue.push(samples)
    case UInt8(ascii: "F"):
        log(String(format: "flush (%.2f s dropped)", playQueue.queuedSeconds))
        playQueue.clear()
    case UInt8(ascii: "P"):
        playQueue.setPaused(true)
        log(String(format: "pause (%.2f s queued)", playQueue.queuedSeconds))
    case UInt8(ascii: "R"):
        playQueue.setPaused(false)
        log(String(format: "resume (%.2f s queued)", playQueue.queuedSeconds))
    case UInt8(ascii: "Q"):
        shutdown(0)
    default:
        log("unknown frame kind \(kind), ignored")
    }
}
