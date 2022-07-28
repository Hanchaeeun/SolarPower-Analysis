# SolarPower-Analysis

태양광 발전량의 영향을 미치는 결정 요소 분석

### load data
![데이터자료](https://user-images.githubusercontent.com/72204267/181611799-ae080631-1a1d-48ce-bb53-c9ff1472b8e2.png)


#### Feature Data
 >+ ASOS (종관기상자료)
 >
 >   기상자료개방포털 https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36
 >  
 >+ 미세먼지 데이터
 > 
 >   에어코리아 https://www.airkorea.or.kr/web/last_amb_hour_data?pMENU_NO=123
 > 

#### target data
>+ 태양광 전력거래량 
>
>   공공데이터포털 https://www.data.go.kr/data/15005796/fileData.do
> 

### Models
1. 머신러닝 모델
3. 딥러닝 (RNN모델)
4. Shap

### process 
1. 데이터 다운로드
2. 데이터 전처리
3. 편차 열 생성
4. 모델 학습
5. Shap을 통한 변수 비교
