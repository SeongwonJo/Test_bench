## About project

pytorch image classification 

<br><br><br>

## Usage

code examples

### Training

```sh
python main.py 
```

- properties.yml 에서 배치사이즈, 에폭 수, 데이터 셋, 모델 종류, 옵티마이저, 학습률 설정할 수 있습니다.

- -d 로 장치 선택 가능, -c 로 결과 csv파일 경로 설정 가능 필요시 뒤에 추가해서 사용

  ```sh
  python main.py -d "cuda:1" -c "D:/folder/result.csv"
  ```

<br><br>

### Inference

- 데이터 세트에 대한 추론 정확도 출력

  1. 테스트할 이미지들이 들어있는 폴더를 argument로 입력

     - 폴더의 안에 이미지 파일이 각 클래스 폴더에 들어가있어야 함

     - ```
       test/
       ├── NORMAL
       │   ├── 1.jpeg
       │   ├── 2.jpeg
       │   └── ...
       │    
       └── PNEUMONIA
           ├── 4.jpeg
           ├── 5.jpeg
           └── ...
       ```

  ```sh
  python inference.py -i './test'
  ```

  * 모델에 입력되는 이미지 크기(resolution), 모델 종류(net), 데이터셋, 사용할 장치, pt 파일의 경로를 inference_settings.yml 파일에서 설정할 수 있습니다.

<br>

- 하나의 이미지에 대한 추론 결과 출력

  1. 테스트할 이미지의 경로를 argument로 입력

  ```sh
  python inference_one_image.py -i './test/BACTERIA-134339-0001.jpeg'
  ```


<br>

- 폴더 안의 모든 이미지에 대한 결과를 dictionary로 출력

  1. 폴더 경로 argument로 입력

  ```sh
  python inference_folder.py -i './test'
  ```


<br>

- test용 pt파일
  - https://etri.gov-dooray.com/share/drive-files/kdhawuidbrbd.xeBDyzezQZOwrAzlOmtSCQ
  - 다운로드 후 inference 코드가 있는 폴더(default 경로)로 이동