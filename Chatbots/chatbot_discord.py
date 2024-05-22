import warnings
from discord import Intents
import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from discord.ext import commands
import os

GREETING_INPUTS = (
    "hello",
    "hi",
    "greetings",
    "sup",
    "what's up",
    "hey",
)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello there", "I am glad! You are talking to me"]

lemmer = nltk.stem.WordNetLemmatizer()
sent_tokens = []
word_tokens = []


def LemTokens(tokens: list[str]) -> list[str]:
    return [lemmer.lemmatize(token) for token in tokens]


def LemNormalize(text: str) -> list[str]:
    return LemTokens(nltk.word_tokenize(text.lower().translate(dict((ord(punct), "") for punct in string.punctuation))))


def greeting(sentence: str):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response: str):
    robo_response = ""
    sent_tokens.append(user_response)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
        tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo_response = "I'm sorry! I don't understand you"
    else:
        robo_response = robo_response + sent_tokens[idx]

    return robo_response


bot = commands.Bot(command_prefix="&", intents=Intents.all())


@bot.event
async def on_ready():
    print(f"We have logged in as: {bot.user}")


@bot.command(name="greet")
async def hello(ctx: commands.context.Context):
    await ctx.send(f"Hello! {ctx.message}")


@bot.command(name="roll_dice", help="Simulates rolling dice.")
async def roll(ctx, number_of_dice=1, number_of_sides=6):
    dice = [str(random.choice(range(1, number_of_sides + 1))) for _ in range(number_of_dice)]
    await ctx.send(", ".join(dice))


@bot.command(name="chat", help="Message with this dummy chat :v")
async def chat(ctx: commands.context.Context):
    user_response = ctx.message.content.removeprefix("&chat ")
    print(f"Processing: {user_response}")
    if user_response == "bye":
        await ctx.send("Bye! take care...")
        return
    if user_response in ["thanks", "thank you", "thx"]:
        await ctx.send("You are welcome...")
    elif user_response in ["idk", "what?", "..."]:
        await ctx.send("I'm just a dummy chatbot...")
    else:
        if (greet := greeting(user_response)) is not None:
            await ctx.send(f"{greet}")
        else:
            await ctx.send(f"{response(user_response)}")
            sent_tokens.remove(user_response)


if __name__ == "__main__":
    load_dotenv()
    nltk.download("punkt")
    nltk.download("wordnet")

    with open("cleaned.txt", "r", errors="ignore") as file:
        raw = file.read().lower()
        sent_tokens = nltk.sent_tokenize(raw)
        word_tokens = nltk.word_tokenize(raw)

    token = os.getenv("DISCORD_TOKEN")
    bot.run(token)
