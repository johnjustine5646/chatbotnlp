import string
import nltk


lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = {ord(punct):None for punct in string.punctuation}

nltk.download('punkt')
nltk.download('wordnet')

# BUILDING SIMPLE CHATBOT USING PYTHON

from sklearn.feature_extraction.text import TfidfVectorizer  # vectorizer
# user for finding cosine similariry
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np  # Used for working with arrays
import random  # To choise random varibles,random choice


import argparse  # To Parse the arguments.
import string  # Contains a number of functions to process standard Python strings
import nltk  # Text processing libraries


# grouping together the different inflected forms of a word
lemmer = nltk.stem.WordNetLemmatizer()
# common punctuations.
remove_punct_dict = {ord(punct): None for punct in string.punctuation}

nltk.download('punkt')
nltk.download('wordnet')


GREET_INPUT = ("hello","hi","sup")
GREET_RESPONSE = ['Hi', "Hello", "I am glad you are talking to me!"]

# Text Preprocessing


class Preprocess:
    """Preprocessing the text data(raw_doc).
    Args:
        raw_doc: Text information you want to use for chatbot.
    """

    def __init__(self, raw_doc):
        self.sent_tokens = nltk.sent_tokenize(raw_doc)

    def lem_normalize(self, text):
        """Normalizing data. Converting text to lower case,
         and removing punctuation from text and tokenizing them to words.

        Args :
            text: The text that use for chatbot(corpus)

        Returns: 
            It returns the grouping together the different inflected forms of a word so they can be analyzed as a single item.

        Example:
            -> rocks : rock
            -> corpora : corpus
            -> better : good
        """
        tokens = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
        lem_token = [lemmer.lemmatize(token) for token in tokens]
        return lem_token

    def greet(self, user_response):
        """Function Used for responding for greeting.

        Args:
            user_response: text/query that user ask/say.
        Returns:
            Choose the random greet from list(GREET_RESPONSE) and returns greet.
        """
        for word in user_response.split():
            if word.lower() in GREET_INPUT:
                return random.choice(GREET_RESPONSE)

    def response(self, user_response):
        """Finds the similarity between the input or given text using cosine_similarity,
            Gives the index of similarity of user_response.

        Args: 
            user_response: text/query that user ask/say.

        Returns:
            similarity found index in text and If user_response is not understandable(where the sim
            ilarity between the user_response is too far. it returns text."
        """
        self.sent_tokens.append(user_response)
        tfidfvec = TfidfVectorizer(
            tokenizer=self.lem_normalize, stop_words='english')
        tfidf = tfidfvec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]

        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        self.sent_tokens.remove(user_response)
        if (req_tfidf == 0):
            return "I am Sorry! I dont understand you"
        else:
            return str(self.sent_tokens[idx])


class ChatBot:
    """Starts the Session, If the requirements are satisfied.
    Args:
        text_path : Text path where your coupus text file exist.

    Returns:
        "Bot Chatting with User"
    """

    def __init__(self, text_path):
        raw_docs = self.load_text(text_path)
        self.prep = Preprocess(raw_docs)

    def load_text(self, text_file_path):
        f = open(text_file_path, "r", errors='ignore')
        raw_doc = f.read()
        raw_doc = raw_doc.lower()
        return raw_doc

    def start_chat(self):
        """Sesion starts when function runs and writes the input and output response in new file named as user name(user_name.txt),
        Returns :
            Bot starts responsing to your queries/anything.
        """
        flag = True
        print("BOT : My Name is BOT, Let's Have Conversation! If you want to exit any time, just type Bye! ")
        print("\n")
        name = input("Please Enter Your Name : ")

        with open(f'{name}.txt', 'a') as ub_chats:
            ub_chats.write(
                "BOT : My Name is BOT, Let's Have Conversation! If you want to exit any time, just type Bye! \n")

        while (flag == True):
            user_response = input(f"{name} : ")
            with open(f'{name}.txt', 'a') as ub_chats:
                ub_chats.write(f"{name} : " + user_response+"\n")
                if (user_response != "bye"):
                    if (user_response == 'thanks'):
                        flag = False
                        ub_chats.write("BOT : You are welcome..\n")
                        print("BOT : You are welcome..")
                    else:

                        if (self.prep.greet(user_response) != None):
                            ub_chats.write(
                                "BOT : " + self.prep.greet(user_response)+f", {name}" + "\n\n")
                            print("BOT :" + "\t"+self.prep.greet(user_response))
                        else:
                            #print("BOT :", end="")
                            ub_chats.write(
                                "BOT : " + self.prep.response(user_response) + "\n\n")
                            print(self.prep.response(user_response))
                    print("\n")

                else:
                    flag = False
                    print("BOT : Goodbye! Take Care")
        with open(f'{name}.txt', 'a') as ub_chats:
            ub_chats.write("\n\n\n")


if __name__ == "__main__":
    """Runs the file automatically,When user runs this script file.
        and adds argument for text_path(corpus).
    """
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-tp", "--txt_path", default="chatbot.txt",
                        help='Give Your Text File Directory.')

    args = parser.parse_args()
    if args.txt_path:
        path = args.txt_path
        chat_bot = ChatBot(path)
        chat_bot.start_chat()
    else:
        raise "Give text file directory"
