def final_prediction(predictions, threshold=0.5):
    final_preds = []
    for pred in predictions:
        votes = {}
        for idx in pred:
            votes[idx] = votes.get(idx, 0) + 1
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        best_match = sorted_votes[0][0] if sorted_votes[0][1] >= threshold * len(pred) else pred[0]
        final_preds.append(best_match)
    return final_preds

if __name__ == "__main__":
    import numpy as np
    # Example usage with dummy data
    #dummy_preds = [[1, 2, 2, 3], [4, 4, 4, 5]]
    final = final_prediction(dummy_preds, threshold=0.6)
    print("Final predictions:", final)
