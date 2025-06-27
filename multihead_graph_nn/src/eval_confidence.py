import math

def softmax_confidence(logits):
    # Shift logits for numerical stability
    max_logit = max(logits)
    shifted_logits = [x - max_logit for x in logits]

    # Exponentiate shifted logits
    exp_logits = [math.exp(x) for x in shifted_logits]


    # Compute softmax probabilities
    sum_exp = sum(exp_logits)
    softmax_probs = [x / sum_exp for x in exp_logits]

    return softmax_probs

conf = softmax_confidence([6.2921, 7.7018, -25.2679, 14.2972])
print("Softmax probabilities:", conf)
print("Top class confidence:", max(conf))

