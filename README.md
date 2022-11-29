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

- 코드 수정 후 업데이트





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
