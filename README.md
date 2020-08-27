# Time Series Forecasting을 통한 RNN, LSTM, GRU 비교

## 1. 서론
딥러닝은 인공지능 기술 중 가장 뛰어난 성능의 알고리즘 중 하나로 알려져 있다. 대표적인 딥러닝 알고리즘에는 CNN, RNN, LSTM, GRU 등이 있다. 

본 프로젝트에서는 이 중에서 RNN, LSTM, GRU을 이용해, 미국 지역의 시간당 전력 소비량 데이터를 학습시켰다. 이렇게 학습된 모델 각각의 정확성, 학습 시간과 같은 성능을 비교, 평가하는 것을 목표 task로 설정하였다.


## 2. 데이터셋, 알고리즘 선택
### 2-1. 데이터셋 선택
과제에서 주어진 데이터셋들은 도시가스나 전력과 같은 에너지 사용량에 대한 데이터이기 때문에, 시간당 에너지 사용량 예측 모델을 만들기로 했다. 이를 위해 측정 간격이 1시간 이하이고 충분히 많은 양의 데이터가 모여있는, ‘미국 지역별 시간당 전력 소비량’ 데이터셋을 선택했다. 모든 지역의 데이터를 사용하면 학습 시간이 오래 걸리기 때문에 임의로 AEP_hourly.csv의 데이터만 사용하기로 했다.

AEP_hourly.csv에는 ‘Datetime’, ‘AEP_MW’ 2개의 정보가 있다. ‘Datetime’은 연월일시에 대한 정보이고, ‘AEP_MW’는 그 Datetime에 따른 AEP에서의 전력 소비량이다.

### 2-2. 알고리즘 선택
선택한 데이터를 학습시키기 위한 적절한 알고리즘을 찾기 위해 먼저 딥러닝 알고리즘에 대한 조사를 간단히 진행하였다. 조사한 알고리즘은 서론에서 언급한 CNN, RNN, LSTM, GRU이다.

CNN(Convolutional Neural Network) 알고리즘은 사람의 시신경 구조를 모방한 구조이다. 기존의 방식은 데이터에서 지식을 추출해 학습이 이루어졌으나, CNN은 데이터를 feature로 추출하여 이 feature들의 패턴을 파악하는 구조이다. 이 CNN 알고리즘은 Convolution 과정과 Pooling 과정을 통해 진행된다. Convolution Layer와 Pooling Layer를 복합적으로 구성하여 알고리즘을 만든다. 이 CNN의 활용 용도는 보통 정보추출, 문장분류, 얼굴인식에 사용된다.

RNN(Recurrent Neural Network) 알고리즘은 반복적이고 순차적인 데이터학습에 특화된 인공신경망의 한 종류로서, 내부의 순환구조가 들어있는 특징이 있다. 순환구조를 이용하여, 과거의 학습을 Weight를 통해 현재 학습에 반영한다. 기존의 지속적이고 반복적이며, 순차적인 데이터학습의 한계를 해결하였다. 현재의 학습과 과거의 학습이 연결 가능해졌으며, 시간에 종속된다. 음성 Waveform을 파악하거나, 텍스트의 문장 앞뒤 성분을 파악할 때 주로 사용된다. RNN의 단점은 처음 시작한 Weight의 값이 점차 학습될수록 상쇄된다는 것이었는데, 이를 보완한 알고리즘이 LSTM 알고리즘이다.

LSTM(Long Short Term Memory Network) 알고리즘은 Cell State라고 불리는 특징층을 하나 더 넣어 Weight를 계속 기억할 것인지 결정하여, Gradient Vanishing의 문제를 해결하였다. 기존 RNN의 경우, 정보와 정보 사이의 거리가 멀면, 초기의 Weight 값이 유지되지 않아 학습능력이 저하된다. LSTM은 과거의 데이터를 계속해서 update 하므로, RNN보다 지속적이다. Cell State는 정보를 추가하거나 삭제하는 기능을 담당한다. LSTM의 장점은 각각의 메모리와 결과값이 컨트롤 가능하다는 점이다. 그러나 메모리가 덮어 씌워질 가능성이 있고, 연산속도가 느리다는 단점을 가지고 있다.

GRU(Gated Recurrent Units) 알고리즘은 LSTM을 변형시킨 알고리즘으로, LSTM은 초기의 weight가 지속적으로 업데이트되었지만, GRU는 Update Gate와 Reset Gate를 추가하여, 과거의 정보를 어떻게 반영할 것인지 결정한다. Update Gate는 과거의 상태를 반영하는 Gate이며, Reset Gate는 현시점 정보와 과거 시점 정보의 반영 여부를 결정한다. GRU의 장점은 연산속도가 빠르며, 메모리가 LSTM처럼 덮어 씌워질 가능성이 없다는 점이다. 그러나 메모리와 결과값의 컨트롤이 불가능하다는 단점을 가지고 있다.

본 프로젝트에서 과거의 값에 기반해 미래의 값을 예측, 즉 time series forecasting을 할 것이기 때문에 CNN보다는 RNN, LSTM, GRU가 적합한 알고리즘이라고 판단했다.


## 3. 소스코드
코드에 대한 자세한 설명은 주석으로 대체했다. 코드는 크게 7개의 섹션으로 되어있는데, 여기서는 각 섹션에서 무엇을 했는가에 대해 대강 설명하겠다.

### 3-1. Basic step
이 섹션에서는 모델 설계를 위한 라이브러리들을 import 했다. 또한, 학습시킬 데이터셋을 load 하기 위해 구글 드라이브와 연동하는 작업도 거쳤다. ‘미국 지역별 시간당 전력 소비량’ 폴더명을 data로 변경하고, 구글 드라이브 최상위에 업로드하면 된다.

### 3-2. Data loading and data exploration
이 섹션에서는 학습에 사용할 AEP_hourly.csv의 데이터를 로드하고 데이터에 대해서 탐색했다. 추가적인 데이터 분석에 앞서, 데이터를 normalize 했다. normalize에는 sklearn MinMaxScaler를 사용했다. sklearn MinMaxScaler는 데이터의 최댓값과 최솟값을 계산해, 원하는 범위의 값으로 normalize 해준다. 아래는 data normalization 이후 데이터를 시각화한 것이다.

### 3-3. Prepare data for training the RNN models
이 섹션에서는 모델 학습을 위한 train data와 test data를 생성했다. 총 121273개의 데이터 중 처음 115000개(약 94.8%)의 데이터를 train data로 설정했고, 나머지 6273개의 데이터를 train data로 설정했다.

### 3-4, 5, 6. Build a simple RNN, LSTM, GRU model
이 섹션에서는 앞서 준비된 데이터들로 simple RNN, LSTM, GRU 모델들을 만들었다. 라이브러리를 사용해 코드를 짰기 때문에 크게 설명할 부분은 없다.
모델들의 성능을 확인하기 위해서 R2 score과 학습 시간 2가지 요소를 확인했다. R2 score는 보통 R2라고 표현되는 coefficient of determination이다. 간단히 말하면 모델의 독립 변수에 의해 설명되는 분산의 비율이라고 할 수 있다. R2 score에 대한 자세한 내용은 Reference에 추가하도록 하겠다. 학습 시간은 단순하게 학습 전후의 시간을 측정해 둘이 빼는 방식으로 계산했다.
다음은 모델별로 측정된 R2 score와 학습 시간이다.

R2 Score of RNN model = 0.9627774160226044
Training Time of RNN model = 77.47840500000001 seconds

R2 Score of LSTM model = 0.9647909940346755
Training Time of LSTM model = 19.627670999999964 seconds

R2 Score of GRU model = 0.9634089347621781
Training Time of GRU model = 18.03053 seconds

### 3-7. Compare predictions made by simple RNN, LSTM, GRU model
단순히 R2 score 점수만 보면, 모델별로 얼마나 정확하게 예측했나 알기 쉽지 않기 때문에, 실제 데이터값과 예측된 데이터값을 시각화해 비교해보았다.


## 4. 결론
먼저, 측정된 simple RNN, LSTM, GRU 각각의 R2 score는 약 0.9627, 0.9647, 0.9634로 유의미한 차이를 찾기 힘들었다. 이는 예상하지 못한 결과였는데, 이론으로 배운 내용대로라면 simple RNN은 Vanishing gradient problem이 발생하여, 데이터가 많은 경우에는 좋은 결과를 내지 못하기 때문이다.
이러한 결과가 나온 이유에 대해서 2가지 가설을 세어보았는데, 첫 번째는 알고리즘을 직접 짠 게 아닌 라이브러리를 사용한 것이므로, 이론적인 simple RNN과는 차이가 있다는 것이다. 이 가설은 아마 아닐 것으로 생각한다. 두 번째는 학습 데이터가, 데이터의 양이나 일관성 등과 같은 면에서 Vanishing gradient problem을 체감할 수 있을 만한 정도가 아닐 수도 있다고 생각했다. 이 부분에 대해서는 추가적인 실험이나 조사가 필요할 것이다.

그리고 학습 시간에서는 유의미한 차이를 찾을 수 있었다. 측정된 simple RNN, LSTM, GRU 각각의 학습 시간은 77.47초, 19.62초, 18.03초였다. 앞에서 조사한 바에 따르면 LSTM은 연산속도가 느리고, GRU는 연산속도가 빠르다고 했는데, simple RNN은 이 둘의 차이를 수십 배 넘을 정도로 느렸다. simple RNN의 학습 시간이 이렇게 압도적으로 오래 걸린 것을 본 것은 예상외였는데, 이것이 위에서 R2 score에 대한 첫 번째 가설을 세우게 된 원인이 되었다.

예측의 정확성 면에서는 크게 차이를 보지 못했더라도 학습 시간에서 유의미한 차이를 보았기 때문에 simple RNN보다는 LSTM과 GRU가 더 나은 알고리즘이라고 결론지을 수 있었다. 사실 단순히 라이브러리를 가져와 사용한 것이므로, 코드를 짜는 데는 큰 차이가 없다. 따라서 이왕 쓸 거라면 LSTM이나 GRU 쓰는 것이 좋을 것이다.


## 5. Reference
http://physics2.mju.ac.kr/juhapruwp/?p=1517
http://dprogrammer.org/rnn-lstm-gru
https://www.kaggle.com/prithvi1029/hourly-energy-consumption-time-series-rnn-lstm
https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru
https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
