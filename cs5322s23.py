import nltk
import codecs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from scipy.spatial.distance import cosine
import gensim.downloader as api
import os

# Load the pre-trained GloVe word embeddings.
model = api.load("glove-wiki-gigaword-200")

# Remove stop words, lemmatize, stem.
def filter_sentence(sentence):
    filtered_sent = []
    filtered_dict = set()
    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)

    for w, pos in tagged_words:
        if pos.startswith('NN') and w not in stop_words:
            new = lemmatizer.lemmatize(ps.stem(w))
            if new not in filtered_dict:
                word_vec = get_vector_or_none(new)
                if word_vec is not None:
                    filtered_sent.append(word_vec)
                filtered_dict.add(new)
                for i in create_sym(w):
                    if i not in filtered_dict:
                        sym_vec = get_vector_or_none(i)
                        if sym_vec is not None:
                            filtered_sent.append(sym_vec)
                        filtered_dict.add(i)

    return filtered_sent


# Add syms for each word
def create_sym(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            synonyms.append(i.name())

    return synonyms


def get_vector_or_none(word):
    try:
        word_vec = model[word]
        return word_vec
    except:
        return None


def check_sim(w1_vec, w2_vec):
    return 1 - cosine(w1_vec, w2_vec)


def WSD_Test_Tissue(sentence):
    target_sent = sentence
    file1 = codecs.open("tissue_organ.txt", 'r', 'utf-8')
    sent1 = file1.read().lower()
    file2 = codecs.open("tissue_paper.txt", 'r', "utf-8")
    sent2 = file2.read().lower()

    filtered_sent1 = filter_sentence(sent1)
    filtered_sent2 = filter_sentence(sent2)
    filtered_target = filter_sentence(target_sent)

    target_1_similarity = 0
    target_2_similarity = 0

    for i in filtered_target:
        for j in filtered_sent1:
            target_1_similarity = target_1_similarity + check_sim(i, j)

        for j in filtered_sent2:
            target_2_similarity = target_2_similarity + check_sim(i, j)

    if target_1_similarity > target_2_similarity:
        return 1
    else:
        return 2


def WSD_Test_Rubbish(sentence):
    target_sent = sentence
    file1 = codecs.open("rubbish_trash.txt", 'r', 'utf-8')
    sent1 = file1.read().lower()
    file2 = codecs.open("rubbish_bull.txt", 'r', "utf-8")
    sent2 = file2.read().lower()

    filtered_sent1 = filter_sentence(sent1)
    filtered_sent2 = filter_sentence(sent2)
    filtered_target = filter_sentence(target_sent)

    target_1_similarity = 0
    target_2_similarity = 0

    for i in filtered_target:
        for j in filtered_sent1:
            target_1_similarity = target_1_similarity + check_sim(i, j)

        for j in filtered_sent2:
            target_2_similarity = target_2_similarity + check_sim(i, j)

    if target_1_similarity > target_2_similarity:
        return 1
    else:
        return 2


def WSD_Test_Yarn(sentence):
    target_sent = sentence
    file1 = codecs.open("yarn_recital.txt", 'r', 'utf-8')
    sent1 = file1.read().lower()
    file2 = codecs.open("yarn_thread.txt", 'r', "utf-8")
    sent2 = file2.read().lower()

    filtered_sent1 = filter_sentence(sent1)
    filtered_sent2 = filter_sentence(sent2)
    filtered_target = filter_sentence(target_sent)

    target_1_similarity = 0
    target_2_similarity = 0

    for i in filtered_target:
        for j in filtered_sent1:
            target_1_similarity = target_1_similarity + check_sim(i, j)

        for j in filtered_sent2:
            target_2_similarity = target_2_similarity + check_sim(i, j)

    if target_1_similarity > target_2_similarity:
        return 1
    else:
        return 2


def main():
    option = input("Choose word:\n1. Rubbish\n2. Yarn\n3. Tissue\n")
    filename = input("Enter the name of the file to process: ")
    lines = []  # Initialize an empty list to store lines
    with open(filename, "r") as file:
        for line in file:
            lines.append(line.strip())

    if option == "1":
        output_filename = "result_rubbish_NickLiu.txt"  # Name of the output file
        if os.path.exists(output_filename):
            os.remove(output_filename)  # Delete the output file if it already exists

        with open(output_filename, "w") as outfile:
            for line in lines:
                result = WSD_Test_Rubbish(line)  # Call your function on the current sentence
                outfile.write(str(result) + "\n")

    elif option == "2":
        output_filename = "result_yarn_NickLiu.txt"  # Name of the output file
        if os.path.exists(output_filename):
            os.remove(output_filename)  # Delete the output file if it already exists

        with open(output_filename, "w") as outfile:
            for line in lines:
                result = WSD_Test_Yarn(line)  # Call your function on the current sentence
                outfile.write(str(result) + "\n")

    elif option == "3":
        output_filename = "result_tissue_NickLiu.txt"  # Name of the output file
        if os.path.exists(output_filename):
            os.remove(output_filename)  # Delete the output file if it already exists

        with open(output_filename, "w") as outfile:
            for line in lines:
                result = WSD_Test_Tissue(line)  # Call your function on the current sentence
                outfile.write(str(result) + "\n")
    else:
        print("Invalid option. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()