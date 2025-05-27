import numpy as np

def post_process(preds):
    new_preds = []
    for pred in preds:
        if pred == -1:
            new_preds.append(np.random.choice(preds))
        else:
            new_preds.append(pred)
    return new_preds

if __name__ == "__main__":
    # Example usage
    #preds = [-1, 1, 2, -1, 4]
    processed = post_process(preds)
    print("Processed preds:", processed)
