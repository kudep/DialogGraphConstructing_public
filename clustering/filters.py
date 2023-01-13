from dataset import Dialogue, Utterance


def speaker_filter(utterance: Utterance, dialogue: Dialogue):
    return utterance.speaker
