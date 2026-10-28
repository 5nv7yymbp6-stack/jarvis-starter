"""Speaker-embedding models behind one interface: embed(int16 mono 16 kHz) ->
unit vector. Everything runs locally. A missing optional model raises
ModelUnavailable rather than failing the whole evaluation."""
import numpy as np

RATE = 16000


class ModelUnavailable(RuntimeError):
    pass


class SpeakerModel:
    name = "base"

    def embed(self, pcm: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def device(self) -> str:
        return "cpu"


class ResemblyzerModel(SpeakerModel):
    """The deployed model: the same 256-d d-vector code path voiceid.py uses."""
    name = "resemblyzer"

    def embed(self, pcm):
        from backtalk import voiceid
        e = np.asarray(voiceid._embed(pcm), float)
        return e / np.linalg.norm(e)


class EcapaModel(SpeakerModel):
    """SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb), 192-d. Downloads the
    public model weights once; no audio ever leaves the machine."""
    name = "ecapa"

    def __init__(self, savedir=None):
        try:
            import torch
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier
        except ImportError as e:
            raise ModelUnavailable(
                f"speechbrain is not installed in this environment ({e})")
        self._torch = torch
        self._clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir) if savedir else None,
            run_opts={"device": "cpu"})

    def embed(self, pcm):
        wav = self._torch.from_numpy(
            np.asarray(pcm, np.float32) / 32768.0).unsqueeze(0)
        with self._torch.no_grad():
            e = self._clf.encode_batch(wav).squeeze().cpu().numpy()
        e = np.asarray(e, float)
        return e / np.linalg.norm(e)


def load(name: str, savedir=None) -> SpeakerModel:
    if name == "resemblyzer":
        return ResemblyzerModel()
    if name == "ecapa":
        return EcapaModel(savedir)
    raise ModelUnavailable(f"unknown model {name!r}")
