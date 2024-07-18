# 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import geopy
from geopy.geocoders import Nominatim
#import googlemaps # 설치하라고 떠서 잠깐만 블락!
import requests
import urllib.request
import urllib.parse
import json
import warnings
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from flask import Flask, render_template, request

app = Flask(__name__)


# filterwarnings 세팅
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# 데이터 불러오기
base_dir = 'C:/Users/sjsms/PY_PROJECTS/dataton/res'

file_path = os.path.join(base_dir, '23.12_전국공장_xy.csv')
file_path_1 = os.path.join(base_dir, '2312_공항항만_xy.csv')
file_path_2 = os.path.join(base_dir, '2308_물류단지터미널창고_xy_2.csv')
file_path_3 = os.path.join(base_dir, '정책지수_최종자료.xlsx')
file_path_4 = os.path.join(base_dir, '4법정동코드_231231_전처리.csv')
file_path_5 = os.path.join(base_dir, '4시도별평균공시지가현황_23.csv')
file_path_6 = os.path.join(base_dir, '5소비자물가_2312.csv')
file_path_7 = os.path.join(base_dir, '5시도별전월세매매_2312.csv')
file_path_8 = os.path.join(base_dir, '5노동생산인구.csv')
file_path_9 = os.path.join(base_dir, '5고속도로출입시설위치정보_2312.csv')
file_path_10 = os.path.join(base_dir, '5국토교통부_전국 버스정류장 위치정보_20231016.csv')
file_path_11 = os.path.join(base_dir, '5한국철도공사_역 위치 정보_20240401.csv')
file_path_12 = os.path.join(base_dir, '5서울시 역사마스터 정보.csv')

data = pd.read_csv(file_path, encoding='UTF-8')
accessibility = pd.read_csv(file_path_1)
warehouse = pd.read_csv(file_path_2, encoding='CP949')
policy_df = pd.read_excel(file_path_3)
code_df = pd.read_csv(file_path_4, encoding='cp949')
std_lot_df = pd.read_csv(file_path_5, encoding='cp949', index_col='2023년기준')
price_1 = pd.read_csv(file_path_6, encoding='cp949')
price_2 = pd.read_csv(file_path_7, encoding='cp949')
population_df = pd.read_csv(file_path_8, encoding='CP949', index_col=0)
ic_df = pd.read_csv(file_path_9, encoding='cp949')
bus_df = pd.read_csv(file_path_10, encoding='cp949')
train_df = pd.read_csv(file_path_11, encoding='cp949')
subway_df = pd.read_csv(file_path_12, encoding='cp949')

# 대표업종 코드의 앞 두자리만 추출
data['대표업종'] = data['대표업종'].astype(str).str[:2]
# 필요한 컬럼만 선택
data = data[['공장주소_지번', '대표업종', 'y', 'x']]
accessibility = accessibility[['유형', '주소지', '공항항구명', '위도', '경도']]
warehouse = warehouse[['유형', '주소', '물류시설명', '위도', '경도']]

# 컬럼 변경
data.columns = ['주소지', '대표업종', '위도', '경도']
accessibility.columns = ['유형', '주소지', '인프라명', '위도', '경도']
warehouse.columns = ['유형', '주소지', '인프라명', '위도', '경도']

# 데이터 결합
combined_df = pd.concat([accessibility, warehouse], ignore_index=True)
# combined_df.to_csv('공항항만물류창고_ xy.csv',encoding='CP949')


GOOGLE_API_KEY = ''
# 주소를 좌표로 변환하는 함수
def geocode_address(address):
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTP 오류가 발생하면 예외를 발생시킵니다.
        data = response.json()
        if data['results']:
            location = data['results'][0]['geometry']['location']
            print(f"주소: {address}, 위도: {location['lat']}, 경도: {location['lng']}")
            return location['lat'], location['lng'] 
        else:
            print(f"주소가 나오지 않는 곳: {address}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"에러발생 주소: {address}")
        print(e)
        return None, None
    

# 하버사인을 통해 거리를 계산
def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # 지구의 반지름 (단위: km)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return distance

# 거리 기반 점수 계산 함수
def calculate_distance_score(distance):
    score = 0
    if distance <= 5:
        score = 5
    elif distance <= 10:
        score = 4
    elif distance <= 30:
        score = 3
    elif distance <= 50:
        score = 2
    elif distance <= 100:
        score = 1
    return score

# 1. 접근성지수
def calculate_accessibility_index(user_x, user_y):
    # 인프라 유형 분류 함수
    def classify_infrastructure(type):
        if '공항' in type:
            return '공항'
        elif '무역항' in type or '신항만' in type:
            return '항만'
        else:
            return '물류창고'

    combined_df['종류'] = combined_df['유형'].apply(classify_infrastructure)

    # 가장 가까운 인프라 계산
    infra_types = ['공항', '항만', '물류창고']
    scores = {}
    for infra_type in infra_types:
        infra_subset = combined_df[combined_df['종류'] == infra_type].copy()
        infra_subset['거리'] = infra_subset.apply(lambda row: haversine(user_x, user_y, row['위도'], row['경도']), axis=1)
        closest = infra_subset.loc[infra_subset['거리'].idxmin()]
        distance = closest['거리']
        scores[infra_type] = calculate_distance_score(distance)
    # 각 인프라의 거리 점수와 가중치 계산
    weights = {'물류창고': 0.7, '공항': 0.1, '항만': 0.2}
    total_score = 0
    for infra_type in infra_types:
        total_score += scores[infra_type] * weights[infra_type]
    return total_score, scores


# 2. 클러스터지수
# 클러스터지수 점수 계산 함수
def calculate_agglomeration_score(count):
    score = 0
    if count == 0:
        score = 0
    elif count <= 1:
        score = 1
    elif count <= 3:
        score = 2
    elif count <= 5:
        score = 3
    elif count <= 10:
        score = 4
    else:
        score = 5

    return score
    
# 클러스터지수 계산 함수 (반경 10km 내)
def calculate_field_agglomeration_count(user_x, user_y, user_industry, data, radius=10):
    relevant_data = data[data['대표업종'].str.startswith(user_industry)]
    tree = KDTree(relevant_data[['위도', '경도']].values)
    radius_in_km = radius
    radius_in_radians = radius_in_km / 6371.0 # km를 라디안으로 변환
    indices = tree.query_radius([[user_x, user_y]], r=radius_in_radians)
    count = len(indices[0]) # 2D 배열에서 인덱스 배열만 추출
    score = calculate_agglomeration_score(count)
    return score


# 3. 정책지수
# 시도명 분류 -> 사용자가 어떻게 시도를 작성할지 알 수 없기 때문
def classify_sido(sido):
        if isinstance(sido, str):
            if '서울' in sido:  return '서울시'
            elif '부산' in sido:    return '부산시'
            elif '대구' in sido:    return '대구시'
            elif '인천' in sido:    return '인천시'
            elif '대전' in sido:    return '대전시'
            elif '광주' in sido:    return '광주시'
            elif '울산' in sido:    return '울산시'
            elif '세종' in sido:    return '세종시'
            elif '경기' in sido:    return '경기도'
            elif '충북' in sido:    return '충청북도'
            elif '충남' in sido:    return '충청남도'
            elif '전남' in sido:    return '전라남도'
            elif '경북' in sido:    return '경상북도'
            elif '경남' in sido:    return '경상남도'
            elif '강원' in sido:    return '강원도'
            elif '전북' in sido or '전라특별' in sido:      return '전라북도'
            elif '제주' in sido:    return '제주도'
        return sido

# 정책지수 반환 함수
def calculate_policy_infrastructure_index(user_address, user_df_row):
    # 사용자 주소를 시도명과 시군구명으로 분리
    address_parts = user_address.split()
    user_sido = address_parts[0]
    user_sgg = address_parts[1]

    # 시도명 변환
    user_sido = classify_sido(user_sido)

    # 정책지수 찾기
    policy_score = policy_df[(policy_df['시도명'] == user_df_row['sido']) & ( (policy_df['시군구명'] == user_df_row['sigungu1']) | (policy_df['시군구명'] == user_df_row['sigungu2']) )]['정책지수'].values
    score = policy_score[0] if len(policy_score) > 0 else 0
    return score


# 4. 부동산지수
# 법정명코드 생성 -> user_df
def make_regionalcode(user_addresses, user_locations):
    user_df = pd.DataFrame({
        '주소': user_addresses, 'longitude': [loc[1] for loc in user_locations], 'latitude': [loc[0] for loc in user_locations],
        'sido': None, 'sigungu1': None, 'sigungu2': None, 'dong': None, 'addr': None, 'regional_code': None  })

    for index, row in user_df.iterrows():
        point = f"{row['longitude']},{row['latitude']}"

        apiurl = "https://api.vworld.kr/req/address?"
        params = {
            "service": "address",
            "request": "getaddress",
            "crs": "epsg:4326",
            "format": "json",
            "type": "PARCEL",
            "point": point,  # x,y 좌표
            "key": '',  # 브이월드 인증키
        }

        response = requests.get(apiurl, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'result' in data['response']:
                parcel_address = data['response']['result'][0]['text']

                parts = parcel_address.split(' ')
                user_df.at[index, 'sido'] = parts[0]
                user_df.at[index, 'sigungu1'] = parts[1]
                user_df.at[index, 'sigungu2'] = ' '.join(parts[1:3])
                user_df.at[index, 'dong'] = parts[-2]
                user_df.at[index, 'addr'] = parts[-1]

                user_df['sido'] = user_df['sido'].apply(classify_sido)
            else:
                print(f"API 응답에서 'result' 키를 찾을 수 없습니다: {data}")
        else:
            print(f"API 요청 실패: {response.status_code}")

    # 법정동코드 찾기
    codes = []
    for idx, row in user_df.iterrows():
        sigungu2 = row['sigungu2'] if isinstance(row['sigungu2'], str) else ""
        matched_rows = code_df[(code_df['시도'] == row['sido']) &
                               ((code_df['시군구'] == row['sigungu1']) | (code_df['시군구'].str.contains(sigungu2))) &
                               (code_df['법정동'] == row['dong'])]

        if not matched_rows.empty:
            regional_code = matched_rows.iloc[0]['법정동코드']
        else:
            regional_code = None

        codes.append(regional_code)

    user_df['regional_code'] = codes
    return user_df if not user_df.empty else pd.DataFrame()

# 부동산지수 도출 함수 - user_df 사용
def get_land_price(user_df):
    user_df['lot_price'] = 0
    for idx, x in enumerate(user_df['regional_code']):
        # 법정동코드 + [1] + 주소지번 + 부번 으로 구성된 고유번호 찾기
        addr_code = user_df.at[idx, 'addr'].split('-')[0]
        code_length = len(str(addr_code))

        if code_length >= 4:
            code = f'{x}1{addr_code}'
        else:
            fill_zeroes = '0' * (4 - code_length)
            code = f'{x}1{fill_zeroes}{addr_code}'

        url = "http://api.vworld.kr/ned/data/getIndvdLandPriceAttr"
        params = {
            "key": '',
            "pnu": code,  # 유저 입력주소의 법정명코드
            "stdrYear": "2023",
            "format": "json",
            "numOfRows": "1000",  # 최대치
            "pageNo": "1"
        }

        # 쿼리 파라미터 인코딩, 요청
        queryParams = "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url + queryParams)
        response_body = urllib.request.urlopen(request).read()

        # json 응답 읽어내고 변환
        response_body = response_body.decode('utf-8')
        response_data = json.loads(response_body)

        # 필드 추출
        land_prices = response_data.get('indvdLandPrices', {}).get('field', [])

        for item in land_prices:
            if item.get('mnnmSlno') == user_df.at[idx, 'addr']:
                user_df.at[idx, 'lot_price'] = float(item.get('pblntfPclnd', 0))
                break

        # 법정동코드 + [2] + 주소지번 + 부번 으로 구성된 고유번호 찾기 (산일 경우)
        if pd.isna(user_df.at[idx, 'lot_price']):
            if code_length >= 4:
                code = f'{x}2{addr_code}'
            else:
                fill_zeroes = '0' * (4 - code_length)
                code = f'{x}2{fill_zeroes}{addr_code}'

            # json 받는 과정 한 번 더
            params['pnu'] = code
            queryParams = "?" + urllib.parse.urlencode(params)
            request = urllib.request.Request(url + queryParams)
            response_body = urllib.request.urlopen(request).read()
            response_body = response_body.decode('utf-8')
            response_data = json.loads(response_body)

            for item in land_prices:
                if item.get('mnnmSlno') == user_df.at[idx, 'addr']:
                    user_df.at[idx, 'lot_price'] = float(item.get('pblntfPclnd', 0))
                    break
    return user_df


# 부동산지수 평가 기준
def lot_score(lot_price, avg_price):
    score = 5
    boundary_1 = avg_price * 1.50 # 최고 기준
    boundary_3_upper = avg_price * 1.1
    boundary_3_lower = avg_price * 0.9
    boundary_5 = avg_price * 0.50 # 최저 기준

    if lot_price >= boundary_1:
        score = 1
    elif boundary_3_upper < lot_price < boundary_1:
        score = 2
    elif boundary_3_lower <= lot_price <= boundary_3_upper:
        score = 3
    elif boundary_5 < lot_price < boundary_3_lower:
        score = 4
    else:
        score = 5
    return score


# 부동산지수 계산 수행 함수
def evaluate_land_prices(user_df, std_lot_df):
  # 주소별 표준지가 가져오기
  user_df = get_land_price(user_df)
  # 지역별 평균 기준가 가져오기
  user_df['avg_price'] = user_df['sido'].map(std_lot_df['평균공시지가'])
  # 부동산지수 계산
  user_df['부동산지수'] = user_df.apply(lambda row:
                                   lot_score(row['lot_price'], row['avg_price']) if row['lot_price'] > 0 else 0, axis=1)
  return user_df

# 5. 노동력지수
# 5-1-1. 정주여건지수
# 5-1-2. 물가지수 전처리 (price_1)
# 유사 컬럼 통합
price_1['버스'] = (price_1['시내버스(성인)_카드']+price_1['시내버스(성인)_현금'])/2
price_1['도시가스'] = (price_1['도시가스가정용_소비자']+price_1['도시가스_도매']+price_1['도시가스_소매'])/3
price_1['상하수도'] = (price_1['상수도가정용']+price_1['하수도가정용'])/2
price_1['이미용'] = (price_1['이용']+price_1['미용'])/2

# 통합으로 필요 없는 컬럼 드랍
price_1.drop(columns=['전철(성인)_카드', '전철(성인)_현금', '시내버스(성인)_카드', '시내버스(성인)_현금', '도시가스가정용_소비자',
                      '도시가스_도매', '도시가스_소매', '상수도가정용', '하수도가정용', '이용', '미용', '삼겹살(환산전)'], inplace=True)

# 시도명 통일
price_1['구분'] = price_1['구분'].apply(classify_sido)
price_1.set_index('구분', drop=True, inplace=True) # 인덱스 처리 (17,17)


# 5-1-2. 전월세지수 전처리 ( price_2 )
price_2['시도명'] = price_2['시도명'].apply(classify_sido) # 시도명 통일 (89, 10)
price_2_mm = price_2[['중위전세(천원)', '중위월세(천원)', '중위보증금(천원)']] # minmax 필요 컬럼만 (89, 3)

# 5-1-3. 생산가능인구지수 추가 전처리 (population_df)
 # 생산가능인구 15~64세, 시도 제외하고 시군구별 비교하도록 스케일링 함수화
def calculate_population(population_df):
    population_df = population_df[(population_df['시군구명'] != '전체') & (population_df['읍면동명'] == '전체')]
    population_df = population_df[population_df['총인구수'] != 0]  #(262, 6)

    scaler = MinMaxScaler() # 인구지수 minmax 적용
    population_df['생산인구비중'] = population_df['생산가능인구'] / population_df['총인구수']
    population_df['생산가능인구지수'] = scaler.fit_transform(population_df[['생산가능인구']])

    # 시군구 단위 비교지수
    pop_score_df = population_df[['시도명', '시군구명', '읍면동명', '생산가능인구지수']].copy()
    pop_score_df['시도명'] = pop_score_df['시도명'].apply(classify_sido)
    return pop_score_df


# minmax 적용, 역계산 함수
def apply_minmax_reverse(df, col_name):
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    reversed_df = (1 - scaled_df).round(10)  # 가격이 높을수록 점수가 낮아지도록 뒤짐음
    tmp = df.copy() # 경고메시지 때문
    tmp[col_name] = reversed_df.mean(axis=1) # 평균값 도출
    return tmp

# 5-2 전월세, 물가지수 구하는 minmax 실행
pr2_reversed = apply_minmax_reverse(price_2_mm, '전월세지수')
pr1_reversed = apply_minmax_reverse(price_1, '물가지수')

# 차례대로 물가, 전월세, 생산가능인구 점수화, 1점 만점
labor1 = pd.DataFrame({ '구분': price_1.index, '물가지수': pr1_reversed['물가지수']}).reset_index(drop=True) #(17,2)
labor2 = pd.DataFrame({ '시도명': price_2['시도명'], '시군구명': price_2['시군구명'], '전월세지수': pr2_reversed['전월세지수']}) # (89, 3)
labor3 = calculate_population(population_df) #(262, 2)

# 5-3. 교통접근성지수 전처리 (ic_df, bus_df, train_df, subway_df)
# 고속도로 IC 컬럼값 변경
ic_df.rename(columns={'X좌표값':'경도', 'Y좌표값':'위도'}, inplace=True)

# 철도, 지하철 합치기 앞서 전처리
train_df = train_df.drop(columns='출입구 개수')
subway_df = subway_df.drop(columns='역사_ID')

train_df = train_df.rename(columns={'지역본부':'분류'})
train_df['분류'] = '철도'
subway_df = subway_df.rename(columns={'호선':'분류', '역사명': '역명'})
subway_df['분류'] = '지하철'

train_merged = pd.concat([train_df, subway_df]) # 합친값
train_merged = train_merged[['분류', '역명', '경도', '위도']]
# 버스정류장 df 전처리
bus_df = bus_df[['정류장명', '도시명', '경도', '위도']]

# 5-3-2. 교통접근성 환산을 위한 기준 설정
bus_radii = [0.5, 1, 1.5, 2, 3] # 좌표값 기준 버스정류장 수 구하는 반경 기준
train_ic_radii = [0.5, 3, 5, 10, 20] # 좌표값 기준 철도, IC 수 구하는 반경 기준

# 지하철 유무에 따라 지역별 가중치 달리 적용
metro_weights = {'bus': 0.8, 'train': 0.2} # 서울, 인천, 경기
non_metro_weights = {'bus': 0.9, 'train': 0.1} # 비수도권

# 점수표
bus_thresholds = {0.5: 5, 1: 4, 1.5: 3, 2: 2, 3: 1} # 500m 내에 정류장이 있으면 5점
train_ic_thresholds = {0.5: 5, 3: 4, 5: 3, 10: 2, 20: 1} # 20km 넘어야만 철도, IC가 있으면 0점

# 좌표 반경 내 인프라 취합 함수
def count_infra_within_radius(user_location, infra_df, radii):
    user_x, user_y = user_location
    counts = {radius: 0 for radius in radii}  # 각 반경별 초기화
    print(f'user_x : {user_x}, user_y : {user_y}, user_location : {user_location}')
    for _, row in infra_df.iterrows():
        distance = haversine(user_x, user_y, row['위도'], row['경도'])
        for radius in radii:
            if distance <= radius:
                counts[radius] += 1
                break
    return counts

# 인프라 접근성 점수 계산 함수
def calculate_infra_score(counts, thresholds):
    score = 0
    for radius, threshold_score in thresholds.items():
        if counts[radius] > 0:
            score = threshold_score
            break
    return score


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        addresses = request.form.getlist('address')
        industries = request.form.getlist('industry')

        user_locations = []
        for address in addresses:
            user_x, user_y = geocode_address(address)
            if user_x is not None and user_y is not None:
                user_locations.append((user_x, user_y))
        print(f'user_location : {user_locations}')
        user_df = make_regionalcode(addresses, user_locations)

        

        if user_df.empty:
            return "Geocoding failed or invalid addresses provided."

        transport_df = pd.DataFrame({
                    '주소': user_df['주소'],
                    'longitude': [loc[1] for loc in user_locations],
                    'latitude': [loc[0] for loc in user_locations],
                    '교통접근성지수': np.nan, '버스점수': np.nan, '철도점수': np.nan, '고속도로점수': np.nan })
        
        for i, user_row in user_df.iterrows():
            user_location = (user_row['latitude'], user_row['longitude'])
            bus_stop_counts = count_infra_within_radius(user_location, bus_df, bus_radii)
            train_counts = count_infra_within_radius(user_location, train_merged, train_ic_radii)
            ic_counts = count_infra_within_radius(user_location, ic_df, train_ic_radii)

            bus_score = calculate_infra_score(bus_stop_counts, bus_thresholds)
            train_score = calculate_infra_score(train_counts, train_ic_thresholds)
            ic_score = calculate_infra_score(ic_counts, train_ic_thresholds)

            if user_row['sido'] in ['서울시', '경기도', '인천시']:
                total_score = ((bus_score / 5) * metro_weights['bus']) + ((train_score / 5) * metro_weights['train'])
            else:
                total_score = ((bus_score / 5) * non_metro_weights['bus']) + ((train_score / 5) * non_metro_weights['train'])

            transport_df.at[i, '버스점수'] = bus_score
            transport_df.at[i, '철도점수'] = train_score
            transport_df.at[i, '고속도로점수'] = ic_score
            transport_df.at[i, '교통접근성지수'] = total_score + (ic_score / 5)  # 최종 교통접근성지수 계산, 일단 2점 만점

        # 5-4-2. 지역별로 분류된 물가, 전월세, 생산가능인구 df에서 점수 찾기
        # 최종 노동력지수 (labor_score_df)
        labor_score_df = pd.DataFrame({'주소': user_df['주소'], '교통접근성지수':transport_df['교통접근성지수'],
                                    '물가지수': np.nan, '전월세지수': np.nan, '정주여건지수':np.nan, '생산가능인구지수': np.nan, })

        # 5-4-3. 주소 기반 지역지수 찾기 함수
        def search_local_score(user_df, labor1, labor2, labor3, labor_score_df):
            for idx, row in user_df.iterrows():
                sido = row['sido']
                sigungu1 = row['sigungu1']
                sigungu2 = row['sigungu2']

                # 물가지수 매칭 / 시도
                labor1_subset = labor1[labor1['구분'] == sido]
                if not labor1_subset.empty:
                    labor_score_df.at[idx, '물가지수'] = labor1_subset.iloc[0]['물가지수']

                # 전월세지수 매칭 / 시도, 시군구
                labor2_subset = labor2[(labor2['시도명'] == sido) & ((labor2['시군구명'] == sigungu1) | (labor2['시군구명'] == sigungu2))]
                if not labor2_subset.empty:
                    labor_score_df.at[idx, '전월세지수'] = labor2_subset.iloc[0]['전월세지수']
                else:
                    labor2_subset = labor2[labor2['시도명'] == sido]
                    labor_score_df.at[idx, '전월세지수'] = labor2_subset.iloc[0]['전월세지수']

                # 생산가능인구지수 매칭 / 시도, 시군구
                labor3_subset = labor3[(labor3['시도명'] == sido) & ((labor3['시군구명'] == sigungu1) | (labor3['시군구명'] == sigungu2))]
                if not labor3_subset.empty:
                    labor_score_df.at[idx, '생산가능인구지수'] = labor3_subset.iloc[0]['생산가능인구지수']
                else:
                    labor3_subset = labor3[labor3['시도명'] == sido]
                    labor_score_df.at[idx, '생산가능인구지수'] = labor3_subset.iloc[0]['생산가능인구지수']

            return labor_score_df


        # 5-4-4. 모두 1점대인 점수 4개, 통합 후 5점 척도화
        #  '노동력지수' = ( 교통접근성지수 + 정주여건 ) (2.5)  + 생산가능인구 (2.5)
        labor_score_df = search_local_score(user_df, labor1, labor2, labor3, labor_score_df)

        # 교통접근성 1.25점으로 환산
        labor_score_df['교통접근성지수'] = labor_score_df['교통접근성지수'].apply(lambda x: (x/2)*1.25 )

        # 정주여건(물가, 전월세) 합쳐서 1.25점으로 환산 ( 50%, 50% )
        labor_score_df['정주여건지수'] = labor_score_df.apply(lambda row: ((row['물가지수'] + row['전월세지수'])/2) * 1.25, axis=1)
        labor_score_df.drop(columns=['물가지수', '전월세지수'], inplace=True) # 불필요 지수 제거

        # 생산가능 2.5점으로 환산
        labor_score_df['생산가능인구지수'] = labor_score_df['생산가능인구지수'].apply(lambda x: x*2.5 )

        # 3-5 최종점수 구하기/ 노동력지수 (labor_score_df)
        labor_score_df['노동력지수'] = labor_score_df['교통접근성지수'] + labor_score_df['정주여건지수'] + labor_score_df['생산가능인구지수']


        results = []
        for i, address in enumerate(addresses):
            user_x, user_y = user_locations[i]
            user_industry = industries[i]

            # 지수 계산
            total_score, closest_infra = calculate_accessibility_index(user_x, user_y)
            agglomeration_count = calculate_field_agglomeration_count(user_x, user_y, user_industry, data)
            policy_score = calculate_policy_infrastructure_index(address, user_df.iloc[i])

            result = {
                '주소': address,
                '업종': user_industry,
                '접근성 지수': total_score,
                '클러스터지수': agglomeration_count,
                '정책지수': policy_score,
                '위도': user_x,
                '경도': user_y
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df[['주소', '업종', '접근성 지수', '정책지수', '클러스터지수']]

        # 부동산지수, 노동력지수
        lotprice_df = evaluate_land_prices(user_df, std_lot_df)
        results_df['부동산지수'] = lotprice_df['부동산지수']
        results_df['노동력지수'] = labor_score_df['노동력지수']

        # 최종 결과 출력
        results_df = results_df[['주소', '업종', '접근성 지수', '정책지수', '부동산지수', '노동력지수', '클러스터지수', '위도', '경도']]


        # 가중치 정의
        weights = {
            '클러스터지수': 0.08985,
            '접근성 지수': 0.12350,
            '노동력지수': 0.42016,
            '정책지수': 0.15665,
            '부동산지수': 0.20984
        }

        # 최종 점수 계산
        results_df['최종 점수'] = (
            results_df['클러스터지수'] * weights['클러스터지수'] +
            results_df['접근성 지수'] * weights['접근성 지수'] +
            results_df['노동력지수'] * weights['노동력지수'] +
            results_df['정책지수'] * weights['정책지수'] +
            results_df['부동산지수'] * weights['부동산지수']
        )

        # 노동력 확보지수와 최종 점수를 소수점 3자리까지 반올림
        results_df['접근성 지수'] = results_df['접근성 지수'].round(3)
        results_df['노동력지수'] = results_df['노동력지수'].round(3)
        results_df['최종 점수'] = results_df['최종 점수'].round(3)

        # 컬럼 순서 재배열
        results_df = results_df[['주소', '업종', '위도', '경도', '접근성 지수', '정책지수', '부동산지수', '노동력지수', '클러스터지수', '최종 점수']]
        sorted_results_df = results_df.sort_values(by='최종 점수', ascending=False)
    
        top_location = sorted_results_df.iloc[0]
        print(f"최종 점수가 가장 높은 후보지는 {top_location['주소']}입니다.")

        # 지도 생성 및 저장
        m = folium.Map(location=[36.5, 127.5], zoom_start=8, width= '100%',height='1000px')
        for idx, row in sorted_results_df.iterrows():
            popup_text = (
                f"주소: {row['주소']}<br>"
                f"업종: {row['업종']}<br>"
                f"접근성 지수: {row['접근성 지수']}<br>"
                f"정책지수: {row['정책지수']}<br>"
                f"부동산지수: {row['부동산지수']}<br>"
                f"노동력지수: {row['노동력지수']}<br>"
                f"클러스터지수: {row['클러스터지수']}<br>"
                f"최종 점수: {row['최종 점수']}"
            )
            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='red' if row['최종 점수'] == results_df['최종 점수'].max() else 'blue')
            ).add_to(m)

        m.save("candidate_locations.html")

        return render_template('map.html', m=m,top_location=top_location)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
