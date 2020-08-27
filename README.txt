data 폴더를 구글 드라이브 최상위에 업로드하면 됩니다.

- 참고용 코드
from google.colab import drive
drive.mount('/content/gdrive/')

data_dir = "/content/gdrive/My Drive/data/"
print(os.listdir(data_dir))