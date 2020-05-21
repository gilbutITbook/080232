import pandas as pd

def preprocess(df):
    # DataFrame 내 결측값을 제거한다
    def remove_missing_values(df):
        df = df.dropna()
        return df

    # 요금 이상치를 제거한다
    def remove_fare_amount_outliers(df, lower_bound, upper_bound):
        df = df[(df['fare_amount'] > lower_bound) & (df['fare_amount'] <= upper_bound)]
        return df

    # 승객 수 이상치를 최빈값으로 대체한다
    def replace_passenger_count_outliers(df):
        mode = df['passenger_count'].mode().values[0]
        df.loc[df['passenger_count'] == 0, 'passenger_count'] = 1
        return df

    # 위도 경도 이상치를 제거한다
    def remove_lat_long_outliers(df):
        # 뉴욕시 경도 범위
        nyc_min_longitude = -74.05
        nyc_max_longitude = -73.75

        # 뉴욕시 위도 범위
        nyc_min_latitude = 40.63
        nyc_max_latitude = 40.85

        # 뉴욕시 반경 내 위치만 남긴다
        for long in ['pickup_longitude', 'dropoff_longitude']:
          df = df[(df[long] > nyc_min_longitude) & (df[long] < nyc_max_longitude)]

        for lat in ['pickup_latitude', 'dropoff_latitude']:
          df = df[(df[lat] > nyc_min_latitude) & (df[lat] < nyc_max_latitude)]
        return df


    df = remove_missing_values(df)
    df = remove_fare_amount_outliers(df, lower_bound = 0, upper_bound = 100)
    df = replace_passenger_count_outliers(df)
    df = remove_lat_long_outliers(df)
    return df


def feature_engineer(df):
    # 연, 월, 일, 요일, 시간 칼럼을 새로 만든다
    def create_time_features(df):
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['hour'] = df['pickup_datetime'].dt.hour
        df = df.drop(['pickup_datetime'], axis=1)
        return df

    # 유클리드 거리를 계산하는 함수
    def euc_distance(lat1, long1, lat2, long2):
        return(((lat1-lat2)**2 + (long1-long2)**2)**0.5)

    # 이동 거리 칼럼을 추가한다
    def create_pickup_dropoff_dist_features(df):
        df['travel_distance'] = euc_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
        return df

    # 공항과 거리 칼럼을 추가한다
    def create_airport_dist_features(df):
        airports = {'JFK_Airport': (-73.78,40.643),
                    'Laguardia_Airport': (-73.87, 40.77),
                    'Newark_Airport' : (-74.18, 40.69)}

        for airport in airports:
            df['pickup_dist_' + airport] = euc_distance(df['pickup_latitude'], df['pickup_longitude'], airports[airport][1], airports[airport][0])
            df['dropoff_dist_' + airport] = euc_distance(df['dropoff_latitude'], df['dropoff_longitude'], airports[airport][1], airports[airport][0])
        return df

    df = create_time_features(df)
    df = create_pickup_dropoff_dist_features(df)
    df = create_airport_dist_features(df)
    df = df.drop(['key'], axis=1)
    return df
