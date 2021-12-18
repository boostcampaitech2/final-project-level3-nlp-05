from konlpy.tag import Komoran

hon_tokens = [word.rstrip('\n') for word in open('komoran_honorific_token.txt', 'r',encoding='utf-8')]
komoran = Komoran()

kor_begin, kor_end = 44032, 55203
jaum_begin,jaum_end = 12593,12622
moum_begin, moum_end= 12623, 12643
chosung_base = 588
jungsung_base = 28

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

def compose(chrs):
    char = chr(
        kor_begin +
        chosung_base * chosung_list.index(chrs[0]) +
        jungsung_base * jungsung_list.index(chrs[1]) +
        jongsung_list.index(chrs[2])
    )
    return char

def decompose(string):
    c = string[-3]
    if not character_is_korean(c):
        return None
    i = ord(c)
    if (jaum_begin <= i <= jaum_end):
        return (c, ' ', ' ')
    if (moum_begin <= i <= moum_end):
        return (' ', c, ' ')

    # decomposition rule
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )    
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])

def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
            (jaum_begin <= i <= jaum_end) or
            (moum_begin <= i <= moum_end))
            

def honorific_token_check(text): # 존댓말 있는지 확인
    cnt = 0
    for i in komoran.pos(text):
        if str(i) in hon_tokens:
            return False
    return True

def change_text(string):
    if string[-2:] != '다.' : return string  # 며, -> 며,
    elif not string[-3].isalpha(): return string[:-2] + "입니다." # 3.8%다. -> 3.8%입니다.
    elif string == "없다." : return "없습니다." # 없다. -> 없습니다.
    elif '이다.' in string: return string[:-3] + '입니다.' # 이다.->입니다.
    c = list(decompose(string)) # ['ㄴ', 'ㅜ', 'ㄴ']
    if ' ' in c: return string[:-2] +'입니다.' # 의미다. -> 의미입니다. (의밉니다도 중에 선택하기)
    if c[2] == 'ㄴ' : # 나눈다. -> 나눕니다. 
        c[2] = chr(ord(c[2])+14) # ㄴ->ㅂ 바꾸는 부분
        if len(string)==3: return compose(tuple(c))+ "니다." # 한다. -> 합니다.
        else: return string[:-3]+compose(tuple(c))+ "니다." # 바란다. -> 바랍니다.
    elif c[2] == 'ㅂ' or c[2] == 'ㅍ': return string[:-2]+ "습니다."
    elif c[2] == 'ㅆ': return string[:-2]+ "습니다." # 했다. 됐다. -> 했습니다. 
    else : return string[:-2] + "입니다." # 나머지 -> ~입니다.
    
    