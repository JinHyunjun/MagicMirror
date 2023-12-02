from bs4 import BeautifulSoup
import requests

html = requests.get('https://weather.naver.com/today/02390132')

soup = BeautifulSoup(html.text, 'html.parser')

data1 = soup.find('div', {'class':'section_center'})

find_address = data1.find('strong', {'class':'location_name'}).text
print('현재 위치: '+find_address)

find_temperature = data1.find('strong', {'class' : 'current'}).text.strip().replace("°", "")
find_temperature = float(find_temperature[find_temperature.index("도")+1:])
print("현재 온도: "+str(find_temperature)+'°C')

find_weather = data1.find('span', {'class' :'weather'}).text
print('현재 날씨: '+find_weather)

# 현재 온도를 정수로 변환합니다.
current_temperature = int(find_temperature)

# 날씨에 따른 색상 추천 코드
def recommend_color(weather):
    if "맑음" in weather:
        color = "하늘색, 노란색"
    elif "구름" in weather:
        color = "회색, 살구색"
    elif "비" in weather:
        color = "초록색, 파랑색"
    else:
        color = "갈색, 남색"

    return color

# 온도와 날씨에 따른 스타일과 색상 추천 코드
def recommend_style(temperature, weather):
    if temperature < 10:
        style = "따뜻한 외투와 두꺼운 스카프, 장갑 소재"
    elif 10 <= temperature < 20:
        style = "자켓, 셔츠, 가디건 같은 계절성 옷"
    elif 20 <= temperature < 30:
        style = "반팔, 얇은 긴팔, 반바지, 면바지 같은 얇은 옷"
    else:
        style = "민소매티, 반바지, 원피스 등 시원한 옷"

    if "비" in weather:
        style += ", 우산이나 방수재질의 옷"

    color = recommend_color(weather)
    style += f". 추천 색상: {color}"
    
    return style

style_recommendation = recommend_style(current_temperature, find_weather)
print("오늘의 추천 스타일: " + style_recommendation)
print()
print()


