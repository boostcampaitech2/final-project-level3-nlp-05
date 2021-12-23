import re

dict_transliteration = {
    '코로나 십구' : '코로나 일구',
    '코로나십구' : '코로나일구',
    '0시' : '0 시',
    'KLAY' : '클레이',
    'NYSE' : '엔와이에스이',
    '5G'   : '파이브지',
    'GDP' : '지디 피',
    'GNP' : '지 앤피',
    'm2': '제곱미터',
    '~' : '에서',
    '$' : '딜러',
    '₩' : '원',
    '€' : '유로',
    '元' : '위안',
    '¥' : '옌',
    'lb' : '파운드'
    # 밝히다, 밝혔는데: 발히다~라고 발음중 
    # 기업명, 주식 관련...? 이런거?
} 

def num2kor(text, decimal = False):
    """ return korean pronunciation of numbers """
    maj_units = ['만', '억', '조', '경']
    kor_digits = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    units = ['']

    for mm in maj_units:
        units.extend(['십', '백', '천']) # 중간단위
        units.append(mm)

    list_text = list(text.split('.')[0])
    list_text.reverse()
    
    str_result = '' # 결과
    num_len_list_amount = len(list_text)
    
    for i in range(num_len_list_amount):
        str_num = list_text[i]
        # 만, 억, 조 단위에 천, 백, 십, 일이 모두 0000 일때는 생략
        if num_len_list_amount >= 9 and i >= 4 and i % 4 == 0 and ''.join(list_text[i:i+4]) == '0000':
            continue
        if str_num == '0': # 0일 때
            if i % 4 == 0: # 4번째자리일 때(만, 억, 조...)
                str_result = units[i] + str_result # 단위만 붙인다
        elif str_num == '1': # 1일 때
            if i % 4 == 0: # 4번째자리일 때(만, 억, 조...)
                str_result = str_num + units[i] + str_result # 숫자와 단위를 붙인다
            else: # 나머지자리일 때
                str_result = units[i] + str_result # 단위만 붙인다
        else: # 2~9일 때
            str_result = str_num + units[i] + str_result # 숫자와 단위를 붙인다
    str_result = str_result.strip() # 문자열 앞뒤 공백을 제거한다 
    
    if str_result == '':
        str_result = '영'

    elif not str_result[0].isnumeric(): # 앞이 숫자가 아닌 문자인 경우
        if str_result[0] not in '십백천만': str_result = '1' + str_result # 1을 붙인다
    
    for i in str_result:
        if i.isnumeric(): str_result = str_result.replace(i, kor_digits[int(i)])
    
    if decimal:
        decimals = text.split('.')[1]
        for num,kor in enumerate(kor_digits):
             decimals = decimals.replace(str(num), kor)
        return str_result+'점'+decimals

    return str_result


def transliterate_text(text):
    """" transliterate text with numbers """
    # numbers with decimal 
    match1 = re.findall(r'\d+\.\d+', text)
    for mat in match1:
        text = re.sub(mat, num2kor(mat, decimal = True), text)
    
    # numbers of 3 digits or greater
    match2 = re.findall(r'\d{3,}', text) # 3 digit or over
    for mat in match2:
        text = re.sub(mat, num2kor(mat), text)
    
    for pre, post in dict_transliteration.items():
        text = re.sub(pre, post, text)

    return text