"""
Demo: a tiny text model that tunes itself and learns from its user.

Shows the three things this project does well:
  1. Zipf-blended n-gram prediction (cheap, instant, offline)
  2. Self-tuning: the model finds its own best Zipf curve
  3. Personalization: the SAME base model, two different users,
     two different personalities emerging from what they type
"""
import tempfile
import os

from colorama import Fore, Style, init

from ZipfNextWordPredictor import ZipfNextWordPredictor

init(autoreset=True)

BASE_CORPUS = """
i really love to read books at night.
i really love to walk in the park.
i really love to cook dinner for my family.
i really love to write short stories.
i really love to listen to music while working.
today i will rest at home and relax.
today i will visit my old friend.
today i will read a new book.
today i will clean the whole house.
today i will call my mother.
the best part of the day is a quiet morning.
the best part of the day is dinner with friends.
the best part of the day is a long walk.
let me help you with that heavy bag.
let me know when you arrive home.
let me show you the new coffee place.
the city is busy but the park is calm.
the city is loud at noon and quiet at night.
my friend and i walk every single evening.
my friend and i talk about everything for hours.
work is busy this week but calm on friday.
the weather is warm and the sky is clear.
the weather is cold so i stay inside today.
she said the new book was worth reading.
she said the coffee there tastes like home.
we finally finished the project late at night.
we finally finished the move on sunday morning.
every sunday i cook a big lunch for everyone.
every sunday i call my family and relax.
a quiet morning makes the whole day better.
a long walk makes every problem feel smaller.
good music makes the work go faster.
the evening sky over the city looks beautiful tonight.
nothing beats a warm dinner after a long day.
nothing beats a quiet night with a good book.
time moves fast when the work is interesting.
"""

HELD_OUT = """
i really love to swim in the sea.
today i will paint the old fence.
the best part of the day is lunch with the team.
let me show you the photos from the trip.
nothing beats a fresh coffee in the morning.
a quiet evening makes the week feel lighter.
"""

CHEF_TEXTS = """
i really love to bake fresh bread.
i really love to cook pasta for guests.
i really love to taste new sauces.
today i will cook a big family meal.
today i will bake a lemon cake.
today i will try a new pasta recipe.
the best part of the day is the dinner rush.
every sunday i cook for the whole family.
nothing beats warm fresh bread out of the oven.
let me show you my new knife.
"""

CODER_TEXTS = """
i really love to write clean python.
i really love to debug tricky bugs.
i really love to refactor old code.
today i will write tests for the api.
today i will debug the login bug.
today i will deploy the new build.
the best part of the day is the first green test.
every sunday i code on my side project.
nothing beats a clean build after a long refactor.
let me show you the new dashboard.
"""

PROMPTS = ["i really love to", "today i will", "nothing beats a"]


def section(title):
    print(f"\n{Fore.CYAN}{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}{Style.RESET_ALL}")


def show_top(model, prompt, k=4, label=""):
    preds = model.predict_next_words(prompt, num_predictions=k)
    words = ", ".join(f"{Fore.GREEN}'{w}'{Style.RESET_ALL} {Fore.MAGENTA}{p:.2f}{Style.RESET_ALL}"
                      for w, p in preds)
    print(f"  {label:<14} {words}")


def main():
    # ---- 1. Train the base model (one shared corpus, everyone starts equal)
    section("1. Train: 36 sentences, milliseconds, zero GPUs")
    model = ZipfNextWordPredictor(corpus_texts=[BASE_CORPUS], n_gram_size=3,
                                  smoothing="beast")
    print(f"  engine: {Fore.YELLOW}{model.smoothing}{Style.RESET_ALL} "
          f"(Modified Kneser-Ney + Zipf-continuation prior)")
    print(f"  vocab: {Fore.YELLOW}{len(model.corpus_freq)}{Style.RESET_ALL} words, "
          f"{Fore.YELLOW}{model.total_words}{Style.RESET_ALL} tokens trained")

    # ---- 2. Self-tuning: the model calibrates its own smoothing
    section("2. Self-tune: sweeping discounts, prior strength, adaptivity")
    before = model.evaluate(HELD_OUT, k=3)
    report = model.tune(HELD_OUT)
    best = report["best"]
    print(f"  before: perplexity {Fore.YELLOW}{before['perplexity']:.0f}{Style.RESET_ALL} "
          f"(default settings)")
    if "zipf_weight" in best:  # legacy engine report
        print(f"  after : perplexity {Fore.GREEN}{best['perplexity']:.0f}{Style.RESET_ALL}  "
              f"<- picked zipf_weight={best['zipf_weight']}, "
              f"profile='{best['interp_profile']}' by itself")
    else:  # beast / KN-family report
        print(f"  after : perplexity {Fore.GREEN}{best['perplexity']:.0f}{Style.RESET_ALL}  "
              f"<- picked discount_scale={best['discount_scale']}, "
              f"zipf_prior={best['zipf_prior_weight']}, "
              f"adaptive={best['adaptive_prior']} by itself")

    # ---- 3. Personalization: same base model, two users, two personalities
    section("3. Personalize: chef vs coder on the SAME base model")

    with tempfile.TemporaryDirectory() as tmp:
        base_file = os.path.join(tmp, "base.pkl")
        model.save(base_file)  # one trained model...

        def new_user():
            return ZipfNextWordPredictor.load(base_file)

        chef = new_user()
        coder = new_user()
        chef.personalization, chef.personalization_halflife = 0.8, 40.0
        coder.personalization, coder.personalization_halflife = 0.8, 40.0

        print(f"\n  Before any learning - everyone gets the same advice:")
        for prompt in PROMPTS[:2]:
            print(f"\n  {Fore.CYAN}\"{prompt} ...\"{Style.RESET_ALL}")
            show_top(model, prompt, label="base model")

        for text in CHEF_TEXTS.strip().split("\n"):
            chef.observe(text)
        for text in CODER_TEXTS.strip().split("\n"):
            coder.observe(text)

        print(f"\n  After 10 sentences each - watch the personalities split:")
        for prompt in PROMPTS:
            print(f"\n  {Fore.CYAN}\"{prompt} ...\"{Style.RESET_ALL}")
            show_top(model, prompt, label="base model")
            a = chef.user_stats()["effective_alpha"]
            show_top(chef, prompt, label=f"chef a={a:.2f}")
            show_top(coder, prompt, label=f"coder a={a:.2f}")

        # ---- 4. Your turn
        section("4. Your turn: type and the model learns YOU (exit to quit)")
        me = new_user()
        me.personalization, me.personalization_halflife = 0.8, 40.0
        while True:
            try:
                line = input(f"\n{Fore.CYAN}You: {Style.RESET_ALL}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in ("exit", "quit"):
                break
            stats = me.observe(line)
            print(f"  {Fore.YELLOW}(learned: {stats['words']} user words, "
                  f"alpha={stats['effective_alpha']:.2f}){Style.RESET_ALL}")
            show_top(me, line, k=5, label="your next:")
            print(f"  {Fore.YELLOW}completion:{Style.RESET_ALL} "
                  f"{me.predict_completion(line, num_words=4, temperature=1.0)}")


if __name__ == "__main__":
    main()
