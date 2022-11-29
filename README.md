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

