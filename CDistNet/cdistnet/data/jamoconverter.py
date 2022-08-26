import re

class JamoConverter:
    """
    Korean Jamo Converter

    target_text =
    """
    def __init__(self):
        self.chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
            'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.jongsung_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
            'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
            'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.vocab = sorted(list(set(self.chosung_list) | set(self.jungsung_list) | set(self.jongsung_list)))

    def text2list(self, text):
        """
        Convert text to jamo list.

        Args:
            text (str) : Input text. Should only include Korean.
        Returns:
            list of [chosung, jungsung, jongsung]
        """
        assert re.sub(r'[^ㄱ-ㅎ|가-힣]+', '', text) == text, text
        jamo_list = []
        for syllable in text:
            code = ord(syllable)
            cho_idx = int((code - 44032) / 588)
            jung_idx = int((code - 44032 - (cho_idx * 588)) / 28)
            jong_idx = int(code - 44032 - (cho_idx * 588) - (jung_idx * 28))
            cho = self.chosung_list[cho_idx]
            jung = self.jungsung_list[jung_idx]
            jong = self.jongsung_list[jong_idx]
            jamo_list += [[cho, jung, jong]]
        return jamo_list

    def list2text(self, jamo_list):
        """
        Convert jamo list to test.

        Args:
            jamo_list (chr) : List of [chosung, jungsung, jongsung]. Jongsung can be empty chr.
        Returns:
            text (str)
        """
        text = ''
        for cho, jung, jong in jamo_list:
            cho_idx = self.chosung_list.index(cho)
            jung_idx = self.jungsung_list.index(jung)
            jong_idx = self.jongsung_list.index(jong)
            syllable = chr(0xac00 + 28 * 21 * cho_idx + 28 * jung_idx + jong_idx)
            text += syllable
        return text

    def str2list(self, jamo_str, delimiter='<del>'):
        """
        Convert string composed of jamo to jamo list.

        Jamo string must have a delimiter to identify syllable sections.
        Certain conditions should be met, and erroneous syllable sections are ignored.

        e.g.[[ㅇ, ㅏ, ㄴ], [ㄴ, ㅕ, ㅇㄴ]]
        """
        jamo_list = []
        syllable_sections = jamo_str.split(delimiter)
        for syllable_section in syllable_sections:
            # Must have at least one jungsong. If there are multiple junsongs, select first one. If can't find any, skip.
            is_jungsung = [pho in self.jungsung_list for pho in syllable_section]
            if not sum(is_jungsung):
                continue
            else:
                jung_pos = is_jungsung.index(True)
                jung = syllable_section[jung_pos]
                # Find closest chosong. If can't find any, skip.
                cho = None
                for pos, pho in enumerate(syllable_section):
                    if pos < jung_pos and pho in self.chosung_list:
                        cho = pho
                if cho is None:
                    continue
                # Find closest jongsong. If can't find any, jongsung = ''
                jong = ''
                for pos, pho in reversed(list(enumerate(syllable_section))):
                    if pos > jung_pos and pho in self.jongsung_list:
                        jong = pho

                jamo_syllable = [cho, jung, jong]
                jamo_list += [jamo_syllable]
        return jamo_list

    def list2str(self, jamo_list, delimiter='<del>'):
        """
        Convert list of jamo to jamo string with syllable delimiter.

        e.g. ㅇㅏㄴ<del>ㄴㅕㅇ
        """
        jamo_str = ''
        for cho, jung, jong in jamo_list:
            cho_jung_jong = ''.join([cho, jung, jong])
            jamo_str += cho_jung_jong + delimiter
        jamo_str = jamo_str[:-len(delimiter)]
        return jamo_str

    def text2str(self, text, delimiter='<del>'):
        """
        Text -> list -> str.
        """
        jamo_list = self.text2list(text)
        jamo_str = self.list2str(jamo_list, delimiter)
        return jamo_str

    def str2text(self, jamo_str, delimiter='<del>'):
        """
        Str -> list -> text.
        """
        jamo_list = self.str2list(jamo_str, delimiter=delimiter) # processed
        text = self.list2text(jamo_list)
        return text

    def list2label(self, jamo_list, delimiter='<del>'):
        """
        Convert list of jamo to label format list, with delimiter.

        e.g. [ㅇ, ㅏ, ㄴ, <del, ㄴ, ㅕ, ㅇ]
        """
        label_list = []
        for cho, jung, jong in jamo_list:
            label_list.extend([cho, jung, jong, delimiter])
        label_list = label_list[:-1]
        return label_list

    def text2label(self, text, delimiter='<del>'):
        """
        Convert text to label format list, with delimiter.
        """
        jamo_list = self.text2list(text)
        label_list = self.list2label(jamo_list, delimiter=delimiter)
        return label_list
        



if __name__ == "__main__":
    text = '안녕너의이름은'
    c = JamoConverter()
    pass
