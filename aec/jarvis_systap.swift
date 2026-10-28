// jarvis-systap: what the Mac itself is playing, for the voice line.
//
// A CoreAudio process tap (macOS 14.2+) on every app's output, read through a
// private aggregate device, downmixed and written out as int16 LE mono 16 kHz.
// The voice line compares what its mic heard with what the Mac was playing at
// that moment and drops lines that are really a video or music on this Mac.
// Probed 2026-09-15: an 8.6 s clip played with afplay was captured at -35 to
// -48 dBFS while it played, silence before it.
//
// usage: jarvis-systap
// stdout: int16 LE mono 16 kHz, continuous
// stderr: "[systap] ready ..." once running, then log lines
// stdin : EOF quits, so the helper never outlives the voice line
import AudioToolbox
import CoreAudio
import Foundation

setvbuf(stderr, nil, _IONBF, 0)
signal(SIGPIPE, SIG_IGN)
let OUT_RATE = 16000.0

func log(_ s: String) { FileHandle.standardError.write(("[systap] " + s + "\n").data(using: .utf8)!) }

func defaultOutputUID() -> String? {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                          mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var dev = AudioDeviceID(0); var sz = UInt32(MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &sz, &dev) == noErr else { return nil }
    var uaddr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceUID,
                                           mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var cf: Unmanaged<CFString>? = nil; var csz = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(dev, &uaddr, 0, nil, &csz, &cf) == noErr, let s = cf else { return nil }
    return s.takeRetainedValue() as String
}

// ---- output: a writer thread, so the audio thread never blocks on the pipe ----
final class Outbox {
    private var pending = Data()
    private let lock = NSLock()
    private let ready = DispatchSemaphore(value: 0)
    func add(_ samples: [Int16]) {
        var d = Data(count: samples.count * 2)
        d.withUnsafeMutableBytes { raw in
            let out = raw.bindMemory(to: Int16.self)
            for i in 0..<samples.count { out[i] = samples[i].littleEndian }
        }
        lock.lock(); pending.append(d); lock.unlock()
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

// ---- downmix + downsample: box-average each output sample's span of input ----
final class Downsampler {
    let step: Double
    var acc = 0.0, accN = 0, pos = 0.0
    init(inRate: Double) { step = inRate / OUT_RATE }
    func push(_ mono: UnsafeBufferPointer<Float>) -> [Int16] {
        var out: [Int16] = []
        out.reserveCapacity(Int(Double(mono.count) / step) + 1)
        for x in mono {
            acc += Double(x); accN += 1; pos += 1
            if pos >= step {
                pos -= step
                let v = max(-1, min(1, acc / Double(accN)))
                out.append(Int16(v * 32767))
                acc = 0; accN = 0
            }
        }
        return out
    }
}

let outbox = Outbox()

let tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
tapDesc.uuid = UUID()
tapDesc.muteBehavior = .unmuted
tapDesc.isPrivate = true
var tapID = AudioObjectID(kAudioObjectUnknown)
let tapErr = AudioHardwareCreateProcessTap(tapDesc, &tapID)
guard tapErr == noErr else { log("could not create the process tap: \(tapErr)"); exit(1) }

var faddr = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
var fmt = AudioStreamBasicDescription(); var fsz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
guard AudioObjectGetPropertyData(tapID, &faddr, 0, nil, &fsz, &fmt) == noErr, fmt.mSampleRate > 0 else {
    log("could not read the tap format"); AudioHardwareDestroyProcessTap(tapID); exit(1)
}
guard let outUID = defaultOutputUID() else { log("no default output device"); AudioHardwareDestroyProcessTap(tapID); exit(1) }

let aggDesc: [String: Any] = [
    kAudioAggregateDeviceNameKey: "Jarvis system tap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outUID]],
    kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true, kAudioSubTapUIDKey: tapDesc.uuid.uuidString]],
]
var aggID = AudioObjectID(0)
let aggErr = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
guard aggErr == noErr else { log("could not create the aggregate: \(aggErr)"); AudioHardwareDestroyProcessTap(tapID); exit(1) }

let down = Downsampler(inRate: fmt.mSampleRate)
let scratch = UnsafeMutablePointer<Float>.allocate(capacity: 1 << 16)
var procID: AudioDeviceIOProcID? = nil
let ioErr = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, nil) { _, inData, _, _, _ in
    for buf in UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData)) {
        guard let m = buf.mData else { continue }
        let ch = max(1, Int(buf.mNumberChannels))
        let frames = min(Int(buf.mDataByteSize) / 4 / ch, 1 << 16)
        let p = m.assumingMemoryBound(to: Float32.self)
        for f in 0..<frames {
            var s: Float = 0
            for c in 0..<ch { s += p[f * ch + c] }
            scratch[f] = s / Float(ch)
        }
        let out = down.push(UnsafeBufferPointer(start: scratch, count: frames))
        if !out.isEmpty { outbox.add(out) }
    }
}
guard ioErr == noErr, AudioDeviceStart(aggID, procID) == noErr else {
    log("could not start capturing: \(ioErr)")
    AudioHardwareDestroyAggregateDevice(aggID); AudioHardwareDestroyProcessTap(tapID); exit(1)
}

func shutdown(_ code: Int32) -> Never {
    AudioDeviceStop(aggID, procID)
    if let p = procID { AudioDeviceDestroyIOProcID(aggID, p) }
    AudioHardwareDestroyAggregateDevice(aggID)
    AudioHardwareDestroyProcessTap(tapID)
    exit(code)
}

Thread.detachNewThread { outbox.writer() }
log("ready rate=\(Int(fmt.mSampleRate)) channels=\(fmt.mChannelsPerFrame) output=\(outUID) -> 16000 mono")

var byte: UInt8 = 0
while read(0, &byte, 1) > 0 {}   // block until the voice line goes away
shutdown(0)
