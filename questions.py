import nltk
import sys
import os
import string
import math
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Create empty files dictionary
    files = dict()

    # Create a list of all files in directory
    list_files = os.listdir(directory)

    # Traverse through each file in list
    for file in list_files:
        # Create path for current file
        path = os.path.join(directory, str(file))
        # Open file
        file_open = open(path, "rb")
        # Read file and hold contents in string named file_read
        file_read = file_open.read()
        # Add string's contents to files dictionary with its name
        files[file] = file_read.decode("utf-8")
        # Close file
        file_open.close()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Tokenize document
    words = nltk.word_tokenize(document)

    # Traverse through "words" list
    word = 0
    while word < len(words):
        # Convert each word to lowercase
        words[word] = words[word].lower()

        # If the word is 100% punctuation, remove the word
        num_punc = 0
        for letter in words[word]:
            if letter in string.punctuation:
                num_punc += 1
        if num_punc == len(words[word]):
            words.remove(words[word])
            continue

        # If the word is one of the stopwords, remove the word
        elif words[word] in nltk.corpus.stopwords.words("english"):
            words.remove(words[word])
            continue

        word += 1

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Create empty idf dictionary
    idf = dict()

    # Record total number of documents
    doc_total = len(documents)

    # Traverse through documents
    for document in documents:
        # Traverse through each word in document's list
        for word in documents[document]:
            # "word_total" keeps track of # of documents containing the word
            word_total = 0
            # Traverse through every document
            for doc in documents:
                # If word is in document, increase word_total
                if word in documents[doc]:
                    word_total += 1
            # Add to idf dictionary the word along with its IDF
            idf[word] = math.log(doc_total/word_total)

    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Create empty tf-idf dictionary
    tf_idf = dict()
    # Traverse through every file
    for file in files:
        # "sum" holds the sum of tf_idf values for all words in the query
        sum = 0
        # Traverse through each word in query
        for word in query:
            # "tf" holds term frequency for each word
            tf = 0
            # Traverse through each word in file
            for word_in_file in files[file]:
                # If word in file is same as word in query, add one to "tf"
                if word == word_in_file:
                    tf += 1
            # Add the product of IDF and TF for the word to "sum"
            sum += (idfs[word] * tf)
        # Add to tf_idf dictionary the sum with its file
        tf_idf[file] = sum

    # Create empty filenames list to hold top n files
    filenames = []

    # Loop over n times
    for num in range(n):
        # Find best file based on max sum
        best_file = max(tf_idf, key=tf_idf.get)
        # Add best file to "filenames"
        filenames.append(best_file)
        # Remove best file from tf_idf
        tf_idf.pop(best_file)

    return filenames


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Create empty sentences_sum dictionary
    sentences_sum = dict()

    # Traverse through each sentence
    for sentence in sentences:
        # Hold sum of idf for each sentence
        sum = 0
        # Traverse through words in query
        for word in query:
            # If the word is in current sentence, add its idf to sum
            if word in sentences[sentence]:
                sum += idfs[word]
        # Add the total sum of the sentence to the dictionary
        sentences_sum[sentence] = sum

    # Create empty best_sentences list
    best_sentences = []

    # Loop over n times
    for num in range(n):
        # Find sentence with max value
        best_sentence = max(sentences_sum, key=sentences_sum.get)
        # Remove best sentence from dictionary
        best_value = sentences_sum.pop(best_sentence)
        # Find second best sentence
        best_sentence2 = max(sentences_sum, key=sentences_sum.get)
        # Check if the values of both sentences are tied
        if sentences_sum[best_sentence2] == best_value:
            # Check if term density of second sentence is greater than first best sentences
            if term_density(query, best_sentence2) > term_density(query, best_sentence):
                # Add back to the dictionary the first best sentence
                sentences_sum[best_sentence] = best_value
                # Add the second-best sentence to best_sentences list
                best_sentences.append(best_sentence2)
                # Remove the second-best sentence from dictionary
                sentences_sum.pop(best_sentence2)
            else:
                best_sentences.append(best_sentence)
        # If term density of original best sentence was greater, then add it to best_sentences
        else:
            best_sentences.append(best_sentence)

    return best_sentences


def term_density(query, sentence):
    """
    Given a query (a set of words) and a sentence (a list of words), calculates and
    returns the "query term density" for the sentence.
    """
    # Set number of common words = 0
    common_words = 0
    # Traverse through every word in query
    for word in query:
        # If the word also appears in the sentence, increase common_words
        if word in sentence:
            common_words += 1

    # Return the # of common words divided by # of words in the sentence
    return common_words/(len(sentence))


if __name__ == "__main__":
    main()
