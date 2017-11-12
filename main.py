import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Run nltk.download() to install required packages 

text_dir = "data/sample.txt"

def main():
	# Read text data
	with open(text_dir) as file:
		data = file.read()

	for sent in sent_tokenize(data):
		print(sent + "\n")

if __name__ == "__main__":
	main()
