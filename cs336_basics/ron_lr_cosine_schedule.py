
"""
Problem (learning_rate_schedule): Implement cosine learning rate schedule with
warmup
Write a function that takes t, αmax , αmin , Tw and Tc , and returns the learning rate αt according to
the scheduler defined above. Then implement [adapters.get_lr_cosine_schedule] and make sure
it passes uv run pytest -k test_get_lr_cosine_schedule.

(Warm-up) If t < Tw , then αt = t/Tw * αmax .

(Cosine annealing) If Tw ≤ t ≤ Tc , then αt = αmin + 1/2(1=cos((t-Tw)/Tc-Tw)* π))(amax-amin)

(Post-annealing) If t > Tc , then αt = αmin .

"""

import math
def lr_cosine_schedule(t, amax , amin , Tw, Tc):
    if t < Tw:
        return t/Tw * amax
    elif t <= Tc:
        return amin + 1/2 * (1 + math.cos( (t-Tw)/(Tc-Tw)*math.pi)) * (amax-amin)
    else:
        return amin

        