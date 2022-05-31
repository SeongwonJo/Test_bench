## Usage

code examples

### Inference

- 데이터 세트에 대한 추론 정확도 출력

  1. 테스트할 이미지들이 들어있는 폴더를 argument로 입력

  ```sh
  python inference.py -i './test'
  ```

  * 모델에 입력되는 이미지 크기(resolution), 모델 종류(net), 데이터셋, 사용할 장치, pt 파일의 경로를 ㅑinference_settings.yml 파일에서 설정할 수 있습니다.

    

- 하나의 이미지에 대한 추론 결과 출력

  1. 테스트할 이미지의 경로를 argument로 입력

  ```sh
  python inference_one_image.py -i './test/BACTERIA-134339-0001.jpeg'
  ```

  

- 폴더 안의 모든 이미지에 대한 결과를 dictionary로 출력

  1. 폴더 경로 argument로 입력

  ```sh
  python inference_folder.py -i './test'
  ```

  

- test용 pt파일
  - https://etri.gov-dooray.com/share/drive-files/kdhawuidbrbd.xeBDyzezQZOwrAzlOmtSCQ