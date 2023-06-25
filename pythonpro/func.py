import chardet
import re


class InputTxt:
    def __init__(self, addr: str):
        self.addr = addr
        self.senArr = []
        self.contentNum = 0
        self.add_sentence_to_instance()

    def get_txt(self):
        try:
            with open(self.addr, 'rb') as f:
                r = f.read()
                char_info = chardet.detect(r)
                return r.decode(char_info['encoding'])
        except FileNotFoundError:
            print(f"File not found: {self.addr}")
        except IOError:
            print(f"IOError occurred while reading file: {self.addr}")
        except Exception as e:
            print(f"An error occurred while reading file: {self.addr}")
            print(str(e))

    def get_all_sentences(self):
        file_txt = self.get_txt()
        prog = re.compile(r'(\w.*?[.?])')
        sen_list = prog.findall(file_txt)
        return sen_list

    def add_sentence_to_instance(self):
        all_sentences = self.get_all_sentences()
        inner_prog = re.compile(r'(\w.*?)[ ,.?]')
        for single_sentence in all_sentences:
            word_list = inner_prog.findall(single_sentence)
            self.contentNum += 1
            self.senArr.append(word_list)


class WordList(InputTxt):
    def __init__(self, addr: str, index: int = 0):
        self.addr = addr
        self.index = index
        super(WordList, self).__init__(addr)

    def get_word_list(self) -> list | None:
        if self.index < self.contentNum:
            index = self.index
            self.index += 1
            return self.senArr[index]
        else:
            return None


class Dictionary:
    def __init__(self):
        self.dictionary = {}
        self.wordNum = 0

    def add_word_list(self, wordList: list):
        for word in wordList:
            if word not in self.dictionary.keys():
                self.dictionary.update({word: self.wordNum})
                self.wordNum += 1



