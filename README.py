import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import matplotlib.pyplot as plt
import librosa.display
from cooley_tukey import custom_stft, custom_istft

# 1. 페이지 설정 및 커스텀 테마 주입 (Rich Aesthetics)
st.set_page_config(
    page_title="AI Audio Denoising Studio",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Premium Design
st.markdown("""
    <style>
    /* 전체 배경 및 폰트 스타일링 */
    .stApp {
        background-color: #0F0F1A;
        color: #E2E2E9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 사이드바 커스텀 */
    section[data-testid="stSidebar"] {
        background-color: #161626 !important;
        border-right: 1px solid #2A2A3F;
    }
    
    /* 카드 컴포넌트 스타일 */
    .premium-card {
        background-color: #17172A;
        border: 1px solid #2B2B47;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* 그라데이션 타이틀 */
    .gradient-title {
        background: linear-gradient(135deg, #4E89FF 0%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-shadow: 0 4px 10px rgba(78, 137, 255, 0.15);
    }
    
    /* 서브 타이틀 */
    .subtitle {
        color: #8C8C9F;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* 오디오 플레이어 감싸는 영역 */
    .audio-block {
        background-color: #1F1F35;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #333355;
        margin-top: 10px;
    }
    
    /* 탭 헤더 스타일링 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #8C8C9F;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #4E89FF !important;
        border-bottom-color: #4E89FF !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. PPT 명세(공식 및 파라미터)를 직접 구현한 Cooley-Tukey FFT 기반 STFT/ISTFT와 결합
def spectral_subtraction(y, sr, noise_duration=0.5, n_fft=2048, hop_length=512, alpha=1.5, beta=0.02):
    # 직접 구현한 custom_stft 사용
    S = custom_stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)
    phase = np.angle(S)
    
    noise_frames = int(noise_duration * sr / hop_length)
    if noise_frames > magnitude.shape[1]:
        noise_frames = magnitude.shape[1]
    
    # 2. 정적 구간 노이즈 프로필 측정
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # 3. PPT 공식 기반 과차감 및 스펙트럼 바닥 연산
    subtracted_magnitude = magnitude - (alpha * noise_profile)
    clean_magnitude = np.maximum(subtracted_magnitude, beta * noise_profile)
    
    # 4. 직접 구현한 custom_istft 사용하여 원래 신호 복원
    clean_S = clean_magnitude * np.exp(1j * phase)
    y_denoised = custom_istft(clean_S, hop_length=hop_length, length=len(y))
    return y_denoised

def spectral_subtraction_minimum_statistics(y, sr, window_duration=1.5, n_fft=2048, hop_length=512, alpha=1.5, beta=0.02, bias_factor=1.5):
    from scipy.ndimage import minimum_filter1d
    
    # 직접 구현한 custom_stft 사용
    S = custom_stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)
    phase = np.angle(S)
    
    window_frames = int(np.ceil((window_duration * sr) / hop_length))
    if window_frames > magnitude.shape[1]:
        window_frames = magnitude.shape[1]
        
    # 각 주파수 대역의 최소 에너지를 잡음으로 추정
    min_noise = minimum_filter1d(magnitude, size=window_frames, axis=1, mode='nearest')
    noise_profile_dynamic = min_noise * bias_factor
    
    # PPT 공식 기반 과차감 및 스펙트럼 바닥 연산
    subtracted_magnitude = magnitude - (alpha * noise_profile_dynamic)
    clean_magnitude = np.maximum(subtracted_magnitude, beta * noise_profile_dynamic)
    
    # 직접 구현한 custom_istft 사용하여 원래 신호 복원
    clean_S = clean_magnitude * np.exp(1j * phase)
    y_denoised = custom_istft(clean_S, hop_length=hop_length, length=len(y))
    return y_denoised

# 3. Streamlit용 시각화 함수
def plot_comparison_st(y_org, y_clean, sr, title="Noise Reduction Comparison"):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True)
    
    # 투명도 및 세부 배경색 커스텀
    fig.patch.set_alpha(0.0)
    for ax in axes.ravel():
        ax.set_facecolor('#17172A')
        ax.spines['bottom'].set_color('#3F3F5F')
        ax.spines['left'].set_color('#3F3F5F')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#A0A0C0')
        ax.yaxis.label.set_color('#A0A0C0')
        ax.xaxis.label.set_color('#A0A0C0')
        ax.title.set_color('#E2E2E9')

    # Waveforms (Top Row)
    librosa.display.waveshow(y_org, sr=sr, ax=axes[0, 0], color='#4E89FF')
    axes[0, 0].set_title("원본 파형 (Original Waveform)", fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel("Amplitude")
    
    librosa.display.waveshow(y_clean, sr=sr, ax=axes[0, 1], color='#00C851')
    axes[0, 1].set_title("잡음 제거 파형 (Denoised Waveform)", fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel("Amplitude")
    
    # 직접 구현한 custom_stft를 이용하여 데시벨 스펙트로그램 시각화
    stft_org = librosa.amplitude_to_db(np.abs(custom_stft(y_org)), ref=np.max)
    stft_clean = librosa.amplitude_to_db(np.abs(custom_stft(y_clean)), ref=np.max)
    
    img1 = librosa.display.specshow(stft_org, sr=sr, x_axis='time', y_axis='linear', ax=axes[1, 0], cmap='magma')
    axes[1, 0].set_title("원본 스펙트로그램 (Original Spectrogram)", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Frequency (Hz)")
    fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")
    
    img2 = librosa.display.specshow(stft_clean, sr=sr, x_axis='time', y_axis='linear', ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title("잡음 제거 스펙트로그램 (Denoised Spectrogram)", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")
    
    plt.suptitle(title, fontsize=14, fontweight='bold', color='#FFFFFF', y=0.98)
    plt.tight_layout()
    return fig

# 4. UI 및 레이아웃 구성
st.markdown('<div class="gradient-title">🎧 AI Audio Denoising Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">PPT 코드 기반 Cooley-Tukey FFT 및 스펙트럼 차감법 공식 적용 오디오 잡음 제거 도구</div>', unsafe_allow_html=True)

# 사이드바 설정 영역
st.sidebar.markdown("### ⚙️ Denoising Settings")
method = st.sidebar.selectbox(
    "Noise Profile Estimation Method",
    ["Static (초반 정적 구간 기반)", "Dynamic (최소 통계법 기반)"],
    help="Static 방식은 오디오 초반의 무음 구간에서 노이즈 프로필을 구하며, Dynamic 방식은 전체 음원 내 윈도우별 최저 에너지를 추적하여 동적으로 구합니다."
)

st.sidebar.markdown("---")

# 공통 파라미터 (PPT 1권 기본값 적용)
alpha = st.sidebar.slider(
    "Over-subtraction Factor (α)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="값이 클수록 더 강력하게 노이즈가 차감되지만 유효 오디오 데이터 손실이 일어날 수 있습니다. (PPT 기본값: 1.5)"
)

beta = st.sidebar.slider(
    "Spectral Floor (β)",
    min_value=0.001,
    max_value=0.1,
    value=0.02,
    step=0.001,
    format="%.3f",
    help="차감 후 남겨둘 최소 배경 노이즈 비율로, 기계적이고 불쾌한 뮤지컬 노이즈 발생을 차단합니다. (PPT 기본값: 0.02)"
)

# 방식별 추가 파라미터
if "Static" in method:
    noise_duration = st.sidebar.slider(
        "Static Noise Duration (sec)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="노이즈 데이터를 얻기 위해 분석할 파일 초반부의 길이입니다."
    )
else:
    window_duration = st.sidebar.slider(
        "Minimum Search Window (sec)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="동적 최소값을 찾을 시간 단위 크기입니다. 음성 사이의 휴지기보다 약간 긴 시간(보통 1.0~2.0초)으로 설정합니다."
    )
    bias_factor = st.sidebar.slider(
        "Bias Compensation Factor",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="윈도우 최소값에서 실제 노이즈 평균 레벨로 보정하기 위한 곱셈 가중치입니다."
    )

# 메인 기능 - 업로더 카드
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "노이즈를 제거할 오디오 파일을 드래그 앤 드롭하거나 선택하세요.",
    type=["wav", "mp3"],
    help="WAV 및 MP3 포맷을 지원합니다."
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # WAV/MP3 로드
    with st.spinner("오디오 파일을 파싱하는 중..."):
        # librosa는 file-like object를 지원하므로 directly load 가능
        y_org, sr = librosa.load(uploaded_file, sr=None)
        
    st.success("오디오 파일 로드 완료!")
    
    # 2단 레이아웃: 처리 및 오디오 재생
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔈 Original Audio")
        st.audio(uploaded_file, format="audio/wav")
        st.info(f"Sampling Rate: **{sr} Hz** | Duration: **{len(y_org)/sr:.2f} seconds**")
        
    # 노이즈 제거 연산
    with st.spinner("자체 구현 Cooley-Tukey FFT 기반 잡음 필터 작동 중..."):
        if "Static" in method:
            y_clean = spectral_subtraction(
                y_org, sr,
                noise_duration=noise_duration,
                alpha=alpha,
                beta=beta
            )
        else:
            y_clean = spectral_subtraction_minimum_statistics(
                y_org, sr,
                window_duration=window_duration,
                alpha=alpha,
                beta=beta,
                bias_factor=bias_factor
            )
            
    with col2:
        st.markdown("#### 🔊 Denoised Audio")
        
        # 메모리 내 WAV 변환 처리
        virtual_file = io.BytesIO()
        sf.write(virtual_file, y_clean, sr, format='WAV')
        virtual_file.seek(0)
        
        st.audio(virtual_file, format="audio/wav")
        
        # 다운로드 버튼 추가
        st.download_button(
            label="💾 Denoised Audio 다운로드",
            data=virtual_file,
            file_name=f"denoised_{uploaded_file.name}",
            mime="audio/wav"
        )

    # 시각화 비교 영역
    st.markdown("---")
    st.markdown("### 📊 Spectrum & Waveform Visual Comparison")
    st.markdown("> **Note**: 하단에 표시되는 모든 주파수 스펙트럼 분석 결과는 우리가 구현한 **Cooley-Tukey FFT**로 직접 변환된 결과입니다.")
    
    with st.spinner("비교 분석 그래프 렌더링 중..."):
        fig = plot_comparison_st(
            y_org, 
            y_clean, 
            sr, 
            title=f"Denoising Comparison Visualizer ({method.split(' ')[0]} Method)"
        )
        st.pyplot(fig)
        
else:
    # 업로드 대기 화면 플레이스홀더
    st.info("💡 사이드바에서 노이즈 제거 파라미터를 조절하고, WAV 또는 MP3 파일을 올려 즉각적인 잡음 필터링 결과를 청취해 보세요.")
