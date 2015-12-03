.PHONY: all clean

all: clean final.pdf

clean:
	rm -f final.pdf

final.pdf: final.md
	pandoc -t beamer -s final.md -o final.pdf
