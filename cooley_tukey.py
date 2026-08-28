import numpy as np

# 고속 푸리에 변환 (FFT) 함수 구현 (Cooley-Tukey FFT 알고리즘, 시간복잡도 O(NlogN))
# PPT 2권의 코드를 원형 그대로 구현한 함수입니다.
def fft_custom(x):
    N = len(x)
    # 기저조건 (재귀호출 종료되기 위한 조건 : 신호의 길이가 1 = 더이상 쪼갤수 없을때)
    if N <= 1:
        return x
    # 짝수 및 홀수 요소 분리 (재귀 호출)
    even = fft_custom(x[0::2])
    odd = fft_custom(x[1::2])
    # 홀수 인덱스 성분에 회전 인자를 곱해 주파수 도메인에서 필요한 변환 수행
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    # 양의 주파수 성분과 음의 주파수 성분을 결합
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# fft_custom을 활용하여 오일러 공식을 통한 역 고속 푸리에 변환 (IFFT) 구현
# 수학적 원리: IFFT(X) = 1/N * conj(FFT(conj(X)))
def ifft_custom(X):
    N = len(X)
    # 1. 입력 주파수 성분에 켤레 복소수 취하기
    X_conj = np.conjugate(X)
    # 2. 순방향 고속 푸리에 변환(FFT) 수행
    fft_res = fft_custom(X_conj)
    # 3. 결과에 다시 켤레 복소수를 취하고 샘플 수 N으로 나누어 원래 진폭 크기 복원
    return np.conjugate(fft_res) / N

def custom_stft(y, n_fft=2048, hop_length=512, window='hann'):
    """
    직접 구현한 fft_custom을 사용하여 시간 영역 신호를 주파수 대역으로 나누는 단시간 푸리에 변환(STFT)
    """
    if window == 'hann':
        win = np.hanning(n_fft)
    else:
        win = np.ones(n_fft)
        
    # 양끝 대칭 반사 패딩 처리
    pad_width = n_fft // 2
    y_padded = np.pad(y, pad_width, mode='reflect')
    
    # 원본 신호 길이를 완벽히 덮을 수 있도록 프레임 수 올림 계산
    num_frames = int(np.ceil(len(y) / hop_length))
    required_len = (num_frames - 1) * hop_length + n_fft
    
    # 패딩된 신호가 부족하면 오른쪽에 0으로 채움
    if len(y_padded) < required_len:
        y_padded = np.pad(y_padded, (0, required_len - len(y_padded)), mode='constant')
        
    S_list = []
    for t in range(num_frames):
        start = t * hop_length
        # 1. 쪼갠 프레임 신호에 한 창 함수 곱하기
        frame = y_padded[start:start + n_fft] * win
        # 2. PPT에서 제공한 Cooley-Tukey FFT 수행
        frame_fft = fft_custom(frame)
        # 3. 대칭 스펙트럼에서 양의 주파수 대역(단측)만 슬라이싱하여 추출
        S_list.append(frame_fft[:n_fft // 2 + 1])
        
    # 세로축이 주파수 빈(Frequency Bin), 가로축이 시간 프레임인 2D 배열로 합성
    return np.column_stack(S_list)

def custom_istft(S, hop_length=512, window='hann', length=None):
    """
    직접 구현한 ifft_custom을 사용하여 주파수 영역 스펙트럼을 시간 신호로 재합성하는 역 단시간 푸리에 변환(ISTFT)
    """
    n_fft = 2 * (S.shape[0] - 1)
    num_frames = S.shape[1]
    
    if window == 'hann':
        win = np.hanning(n_fft)
    else:
        win = np.ones(n_fft)
        
    expected_padded_len = (num_frames - 1) * hop_length + n_fft
    y_reconstructed = np.zeros(expected_padded_len)
    window_sum = np.zeros(expected_padded_len)
    
    for t in range(num_frames):
        # 각 프레임별로 단측 스펙트럼을 대칭적인 양측 스펙트럼으로 양방향 대칭 복원
        s_frame = S[:, t]
        s_full = np.zeros(n_fft, dtype=complex)
        s_full[:n_fft // 2 + 1] = s_frame
        for k in range(1, n_fft // 2):
            s_full[n_fft - k] = np.conjugate(s_frame[k])
            
        # 직접 구현한 IFFT 구동
        frame_time = ifft_custom(s_full)
        frame_real = np.real(frame_time)
        
        # 중첩 가산(Overlap-Add) 연산
        start = t * hop_length
        y_reconstructed[start:start + n_fft] += frame_real * win
        window_sum[start:start + n_fft] += win ** 2
        
    # 창 함수 중첩 에너지 분배 보정
    window_sum_safe = np.where(window_sum > 1e-10, window_sum, 1.0)
    y_reconstructed /= window_sum_safe
    
    # 패딩 자르기 및 원본 길이로 맞추기
    pad_width = n_fft // 2
    if length is not None:
        y_final = y_reconstructed[pad_width:pad_width + length]
    else:
        y_final = y_reconstructed[pad_width:-pad_width]
        
    return y_final

def self_test():
    """
    Cooley-Tukey FFT & STFT 완벽 복원 수학적 일치 여부 자체 검증 테스트
    """
    print("=== [테스트 시작] PPT 쿨리-튜키 FFT 및 복원도 자체 검사 ===")
    
    sr = 8000  # 빠른 테스트를 위해 샘플링 레이트를 8kHz로 설정
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y_test = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    
    # 테스트 1: NumPy FFT와 직접 만든 fft_custom 결과 오차 비교
    test_frame = y_test[:1024] # 1024개 샘플
    custom_res = fft_custom(test_frame)
    numpy_res = np.fft.fft(test_frame)
    
    diff = np.max(np.abs(custom_res - numpy_res))
    print(f"  - FFT 함수 수치적 차이 최댓값: {diff:.2e}")
    if diff < 1e-10:
        print("  - ✅ FFT 결과 일치 성공!")
    else:
        print("  - ❌ FFT 수치 검증 실패!")
        
    # 테스트 2: custom_stft -> custom_istft 신호 복원 검증
    S = custom_stft(y_test, n_fft=1024, hop_length=256)
    y_rec = custom_istft(S, hop_length=256, length=len(y_test))
    
    rec_diff = np.max(np.abs(y_test - y_rec))
    print(f"  - 음원 재합성 오차 최댓값: {rec_diff:.2e}")
    if rec_diff < 1e-10:
        print("  - ✅ 완벽 복원 성공!")
    else:
        print("  - ❌ 복원 시 손실 발생!")
    print("=== [테스트 완료] ===")

if __name__ == "__main__":
    self_test()
