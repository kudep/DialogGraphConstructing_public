from collections import Counter


def print_clustering(clustering, dialogues, max_utts=10):
    for i in range(clustering.get_nclusters()):
        cluster = clustering.get_cluster(i)

        print(f"Cluster #{i}: {len(cluster)} utterances")
        utt_step = (len(cluster) + max_utts - 1) // max_utts
        if utt_step > 0:
            print('\n'.join([dialogues.utterances[utt] for utt in
                             cluster.utterances[::utt_step]]))
            print()
            print(Counter(
                [dialogues.get_utterance_by_idx(utt).speaker for utt in
                 cluster]))
        else:
            print('NO UTTERANCES')
        print('\n')
