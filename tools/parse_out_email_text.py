from nltk.stem.snowball import SnowballStemmer
import string


def parseOutText(file):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """
    data = []
    for line in file.readlines():
        word_list = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("\"", "").replace(
            ":", "").replace('\n', '').replace('<', '').replace('>', '').replace('!', '').replace("[", '')\
            .replace(']', '').strip().split()
        data.append(' '.join(word_list))
    # data = [item for sublist in data for item in sublist]
    data = ' '.join(data)
    content = data.split("X-FileName")
    words = []
    if len(content) > 1:
        # project part 2: comment out the line below
        # words = text_string
        # split the text string into individual words, stem each word,
        # and append the stemmed word to words (make sure there's a single
        # space between each stemmed word)
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(i) for i in content[-1].split()]
    return ' '.join(words)


def main():
    ff = open("..\maildir/jones-t/all_documents/11004", 'r')
    # ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)


if __name__ == '__main__':
    main()
