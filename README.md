## About project

chest x-ray image classfication with PyTorch

<br><br><br>

## Usage

code examples

### Training

```sh
python main.py 
```

- train_options.yml 에서 배치사이즈, 에폭 수, 데이터 셋, 모델 종류, 옵티마이저, 학습률 설정할 수 있습니다.

- -d 로 장치 선택 가능, -c 로 결과 csv파일명 설정 가능 필요시 뒤에 추가해서 사용

- 결과는 runs/exp{} 에 저장됩니다.

  ```sh
  python main.py -d "cuda:1" -c "result.csv"
  ```

- classification_settings.py 에서 데이터 경로, augmentation 설정, scheduler 사용 등 수정 가능

<br><br>

### Inference

- __Arguments__

  * -y 로 학습할때 사용한 옵션 yaml 파일 입력필수 (기본값: train_options.yml)

  * -i 로 추론할 이미지 경로 설정

  * -o 로 옵션설정 (all, one, onebyone)

    

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
  python inference.py -i './test' -o 'all'
  ```




- 하나의 이미지에 대한 추론 결과 출력

  1. 테스트할 이미지의 경로를 argument로 입력

  ```sh
  python inference_one_image.py -i './test/BACTERIA-134339-0001.jpeg' -o 'one'
  ```

<br>

- 폴더 안의 모든 이미지에 대한 결과를 dictionary로 출력

  1. 폴더 경로 argument로 입력

  ```sh
  python inference_folder.py -i './test' -o 'onebyone'
  ```

<br>

- 입력되는 데이터셋은 256x256 크기 이미지로 변환됨

- 가독성 개선 예정 ..

  

### Etc

- 4k 해상도의 이미지를 사용하면 data loader에서 병목이 걸리므로 미리 이미지 크기를 조정한 데이터셋을 만들고 학습을 진행하는 것을 추천

- img_resize.py 사용법

  - -p 로 데이터셋 경로 지정 필요

  - 다음과 같이 구성되어 있어야함

    ```
    dataset/ (입력한 경로)
    ├── class A
    │   ├── image1
    │   ├── image2
    │   └── ...
    │    
    └── class B
        ├── image4
        ├── image5
        └── ...
    ```

  - 코드 사용 예시

    ```
    python img_resize.py -p "D:/dataset/"
    ```

    - 결과가 "기존폴더명(resize)" 에 저장됨
