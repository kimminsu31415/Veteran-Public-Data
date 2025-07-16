import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("🚀 보훈의료 수요-자원 최적화 분석 시작!")
print("="*50)

# 1단계: API 데이터 수집
print("\n📊 API 데이터 수집 중...")

# API 키 및 URL 설정
api_key = "ew65gD84m6ycBehHf%2F8ZhfO%2BiQ0FPjNd%2F%2B%2BSLgE2rJ6Nf%2FgD%2B8YrPCkkGaVrOcSLZh8sOo3IFGVKmEFmTdGUdg%3D%3D"

# 지역별 보훈대상자 현황 API
demographics_url = f"https://api.odcloud.kr/api/15128684/v1/uddi:8b8ea82c-9c37-4b92-8440-b5ad7e83b0a9?page=1&perPage=50000&serviceKey={api_key}"

# 보훈의료 위탁병원 현황 API  
hospitals_url = f"https://api.odcloud.kr/api/15128685/v1/uddi:32b97d80-8a81-4e5e-8b5b-b73adfb16175?page=1&perPage=1000&serviceKey={api_key}"

# 전국 보훈대상자 현황 API
national_url = f"https://api.odcloud.kr/api/15128683/v1/uddi:0a9ab957-a9b4-4b56-9dc8-eff88e9bcfbc?page=1&perPage=1000&serviceKey={api_key}"

try:
    # 지역별 보훈대상자 데이터
    demographics_response = requests.get(demographics_url)
    demographics_data = demographics_response.json()['data']
    
    # 병원 데이터 
    hospitals_response = requests.get(hospitals_url)
    hospitals_data = hospitals_response.json()['data']
    
    print(f"✅ 지역별 보훈대상자 데이터: {len(demographics_data)}건")
    print(f"✅ 위탁병원 데이터: {len(hospitals_data)}건")
    
except Exception as e:
    print(f"❌ API 호출 오류: {e}")
    print("📂 로컬 CSV 파일을 사용합니다...")
    
    # 로컬 파일 읽기
    demographics_df = pd.read_csv('국가보훈부_국가보훈대상자 시군구별 대상별 성별 연령1세별 인원현황_20241231.csv', encoding='cp949')
    hospitals_df = pd.read_csv('국가보훈부_보훈의료 위탁병원 현황_20250101.csv', encoding='cp949')

# 2단계: 데이터 전처리
print("\n🔧 데이터 전처리 중...")

# 지역별 보훈대상자 집계
age_columns = [str(i) + '세' for i in range(0, 101)]
age_columns = [col for col in age_columns if col in demographics_df.columns]

if not age_columns:
    age_columns = [col for col in demographics_df.columns if col.replace('세', '').isdigit()]

demographics_df['보훈대상자수'] = demographics_df[age_columns].sum(axis=1)

# 시도, 시군구별 집계
region_summary = demographics_df.groupby(['시도명', '시군구명'])['보훈대상자수'].sum().reset_index()
region_summary.columns = ['시도', '시군구', '보훈대상자수']

# 병원 데이터 전처리
hospitals_df['병상수'] = pd.to_numeric(hospitals_df['병상수'], errors='coerce').fillna(0)
hospital_summary = hospitals_df.groupby(['광역시도명', '시군구명'])['병상수'].sum().reset_index()
hospital_summary.columns = ['시도', '시군구', '병상수']

# 데이터 병합
merged = pd.merge(region_summary, hospital_summary, on=['시도', '시군구'], how='left')
merged['병상수'] = merged['병상수'].fillna(0)
merged['천명당_병상수'] = (merged['병상수'] / merged['보훈대상자수'] * 1000).replace([np.inf, -np.inf], 0)
merged = merged[merged['보훈대상자수'] > 0]

print(f"✅ 병합된 데이터: {len(merged)}개 지역")

# 3단계: 클러스터링 분석
print("\n🎯 K-means 클러스터링 수행 중...")

features = ['보훈대상자수', '병상수']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged[features])

kmeans = KMeans(n_clusters=4, random_state=42)
merged['클러스터'] = kmeans.fit_predict(scaled_features)

print("✅ 4개 클러스터로 지역 분류 완료")

# 4단계: 개선된 인터랙티브 대시보드 생성
print("\n📱 개선된 인터랙티브 대시보드 생성 중...")

# 레이아웃 개선: 각 차트를 독립적인 영역으로 분리
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "📊 의료 접근성 분포", "🌍 클러스터별 지역 분포", 
        "🏆 상위/하위 지역 비교", "💡 의료자원 효율성 분석",
        "📈 효율성 지수 상세 분석", "🎯 투자 우선순위"
    ],
    specs=[
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}], 
        [{"colspan": 2, "secondary_y": False}, None]  # 마지막 행은 전체 폭 사용
    ],
    vertical_spacing=0.12,  # 행간 간격 증가
    horizontal_spacing=0.08  # 열간 간격 증가
)

# 1행 1열: 히스토그램
fig.add_trace(
    go.Histogram(
        x=merged['천명당_병상수'], 
        nbinsx=30, 
        name='의료 접근성 분포',
        marker_color='lightblue', 
        opacity=0.7,
        showlegend=False
    ),
    row=1, col=1
)

# 1행 2열: 클러스터별 산점도 
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
cluster_names = ['대도시형', '거점도시형', '농촌균형형', '취약지역형']

for i in range(4):
    cluster_data = merged[merged['클러스터'] == i]
    fig.add_trace(
        go.Scatter(
            x=cluster_data['보훈대상자수'], 
            y=cluster_data['병상수'],
            mode='markers', 
            name=f'{cluster_names[i]}',
            marker=dict(color=colors[i], size=8, opacity=0.7),
            text=cluster_data['시도'] + ' ' + cluster_data['시군구'],
            hovertemplate='<b>%{text}</b><br>보훈대상자: %{x:,}명<br>병상수: %{y:,}개<extra></extra>'
        ),
        row=1, col=2
    )

# 2행 1열: 상위/하위 지역 비교 (버튼으로 토글 가능하게)
top_regions = merged.nlargest(10, '천명당_병상수')
bottom_regions = merged.nsmallest(10, '천명당_병상수')

fig.add_trace(
    go.Bar(
        x=top_regions['시군구'], 
        y=top_regions['천명당_병상수'],
        name='상위 10개 지역',
        marker_color='green',
        opacity=0.8,
        visible=True
    ),
    row=2, col=1
)

fig.add_trace(
    go.Bar(
        x=bottom_regions['시군구'], 
        y=bottom_regions['천명당_병상수'],
        name='하위 10개 지역', 
        marker_color='red',
        opacity=0.8,
        visible=False  # 초기에는 숨김
    ),
    row=2, col=1
)

# 2행 2열: 효율성 분석 (크기 조정)
merged['효율성_지수'] = merged['병상수'] / (merged['보훈대상자수'] + 1) * 1000

fig.add_trace(
    go.Scatter(
        x=merged['보훈대상자수'], 
        y=merged['효율성_지수'],
        mode='markers', 
        name='효율성 지수',
        marker=dict(
            color=merged['천명당_병상수'], 
            colorscale='Viridis', 
            size=8,  # 크기 고정
            opacity=0.7,
            colorbar=dict(title="천명당 병상수", x=1.02)
        ),
        text=merged['시도'] + ' ' + merged['시군구'],
        hovertemplate='<b>%{text}</b><br>보훈대상자: %{x:,}명<br>효율성지수: %{y:.2f}<extra></extra>',
        showlegend=False
    ),
    row=2, col=2
)

# 3행: 전체 폭 사용 - 효율성 지수 상세 분석 (세로 막대그래프)
sorted_regions = merged.nlargest(20, '효율성_지수')

fig.add_trace(
    go.Bar(
        x=sorted_regions['시군구'],
        y=sorted_regions['효율성_지수'],
        name='효율성 지수 TOP 20',
        marker_color='purple',
        opacity=0.8,
        visible=True,
        text=sorted_regions['효율성_지수'].round(2),
        textposition='outside'
    ),
    row=3, col=1
)

# 레이아웃 업데이트 (겹침 방지)
fig.update_layout(
    title=dict(
        text="🏥 보훈의료 자원배분 종합 분석 대시보드 (개선판)",
        x=0.5,
        font=dict(size=20, family="Malgun Gothic")
    ),
    height=1200,  # 높이 증가
    width=1400,   # 폭 증가  
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.1,
        xanchor="center",
        x=0.5
    ),
    font=dict(family="Malgun Gothic", size=11),
    margin=dict(l=80, r=120, t=100, b=150)  # 여백 증가
)

# 버튼 추가 (상위/하위 지역 토글)
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": [True, True, True, True, True, True, False, True, True]}],
                    label="상위 지역",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [True, True, True, True, True, False, True, True, True]}],
                    label="하위 지역", 
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.25,
            xanchor="left",
            y=0.65,
            yanchor="top"
        ),
    ]
)

# 축 레이블 업데이트
fig.update_xaxes(title_text="천명당 병상수", row=1, col=1)
fig.update_yaxes(title_text="지역 수", row=1, col=1)

fig.update_xaxes(title_text="보훈대상자수 (명)", row=1, col=2)
fig.update_yaxes(title_text="병상수 (개)", row=1, col=2)

fig.update_xaxes(title_text="지역", row=2, col=1)
fig.update_yaxes(title_text="천명당 병상수", row=2, col=1)

fig.update_xaxes(title_text="보훈대상자수 (명)", row=2, col=2)
fig.update_yaxes(title_text="효율성 지수", row=2, col=2)

fig.update_xaxes(title_text="지역", row=3, col=1)
fig.update_yaxes(title_text="효율성 지수", row=3, col=1)

# 개선된 HTML 파일 저장
fig.write_html('보훈의료_종합대시보드_개선판.html')
print("✅ 개선된 대시보드 생성 완료: 보훈의료_종합대시보드_개선판.html")

# 기본 차트도 표시
fig.show()

print(f'\n🎯 주요 개선사항:')
print(f'   • 레이아웃 겹침 문제 해결 (높이 1200px로 증가)')
print(f'   • 각 차트 영역 간격 확대 (12% 수직, 8% 수평)')
print(f'   • 효율성 지수 차트를 별도 전체폭 영역으로 분리')
print(f'   • 상위/하위 지역 토글 버튼 위치 조정')
print(f'   • 마커 크기 고정으로 클릭 용이성 개선')
print(f'   • 여백 및 범례 위치 최적화')

print(f'\n📁 생성 파일:')
print(f'   • 보훈의료_종합대시보드_개선판.html')
print(f'\n✨ 레이아웃 개선 완료!') 