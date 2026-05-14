import tenacity
import random

# @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=1, max=4) + tenacity.wait_random(0, 2))
@tenacity.retry(wait=tenacity.wait_exponential_jitter(initial=1, max=30))
def unreliable_function():
    rando = random.randint(0, 10)
    print(rando)

    if rando > 2:
        raise Exception("Shit is broken. Aaahh.")
    else:
        return "Success"

print(unreliable_function())
