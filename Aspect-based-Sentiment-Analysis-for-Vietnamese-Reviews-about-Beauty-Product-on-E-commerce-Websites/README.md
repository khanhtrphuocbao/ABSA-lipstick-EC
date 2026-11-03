# Vietnamese Beauty Review ABSA

Phân tích khía cạnh và cảm xúc trong review son môi tiếng Việt trên Shopee.

Ví dụ: **“Màu son đẹp nhưng giao hàng chậm”** → `COLOUR:Positive`, `SHIPPING:Negative`.

Pipeline thực nghiệm hiện tại so sánh **TF-IDF + Logistic Regression**, **BiGRU** và **PhoBERT**. Mỗi họ có hai cấu hình và ba seed; cấu hình được chọn trên validation là TFIDF-02, BIGRU-02 và PHOBERT-02.

## Cấu trúc

```text
src/
  absa_protocol.py          Quy ước nhãn và phép chấm đã khóa
  evaluate_predictions.py   Chấm prediction theo protocol
  training/                Data loaders, preprocessing, models và training
    models/                TF-IDF/LR, BiGRU/Conv1D, PhoBERT
  testing/                 Lựa chọn checkpoint và runner test có kiểm soát
  analysis/
    common.py              Đọc/ghi CSV và artifact dùng chung
    figure_status.py       Kiểm tra hình còn khớp nguồn hay không
    test02b/               Phân tích định lượng và so sánh từng cặp nhãn
    test02c/               Phân tích lỗi, ví dụ review và diagnostic subsets
    test02d/               Nguồn dữ liệu và renderer của bộ hình đã công bố
    test02d_reporting.py    Tổng hợp manuscript từ các bảng và hình
scripts/                   Tiện ích kiểm tra và khôi phục payload
notebooks/                 Notebook chạy trên Colab
protocols/                 Protocol và cấu hình thực nghiệm
reports/
  experiments/             Kết quả và phân tích thực nghiệm
  report_01_uit/            Bản thảo LaTeX, hình và bản PDF ghép
absa-runs-final-test/       Dự đoán và metrics test đã có
tests/                    Unit tests và regression tests
```

Trong `test02b` và `test02c`, `__main__.py` chỉ xử lý CLI, còn `pipeline.py` chứa các bước phân tích có tên riêng. `common.py` không phụ thuộc mô hình hoặc evaluator. Các import cũ được giữ để tương thích.

## Các lệnh thường dùng

Chạy từ thư mục gốc repository. Các lệnh analysis đọc prediction đã có, không tải model weights hoặc chạy inference.

```sh
# Xem cách gọi, không thực thi phân tích
python3 -m src.analysis.test02b --help
python3 -m src.analysis.test02c --help

# Phân tích định lượng từ prediction đã lưu
python3 -m src.analysis.test02b

# Kiểm tra tình trạng bộ hình hiện có
python3 -m src.analysis.test02d --check-stale

# Biên dịch bản thảo UIT trong một bản sao template tạm
python3 -B reports/report_01_uit/build_preview.py \
  --template-root '../master_thesis copy' \
  --bib 'reports/report_01_uit/absa_chapter2_references (8).bib'
```

Các artifact thực nghiệm được lưu riêng: cần có `absa-runs-final-test/`, `reports/experiments/test_01/` và `reports/experiments/test_02/` để chạy phân tích và integration tests. Validation aggregate nằm ngoài repository; runner tạo hình nhận đường dẫn qua `--validation-aggregate`. Checkpoint và bộ kết quả không nằm trong commit refactor.

Các bước phân tích giữ nguyên file có nội dung giống nhau và từ chối ghi đè file khác nội dung. Không cần chạy lại TEST-02C hoặc closure để đọc những kết quả đã hoàn tất. Xem trực tiếp trong `reports/experiments/test_02/`.

## Môi trường và kiểm tra

- Training dùng các phiên bản trong `protocols/experiments/modeling_requirements_v1.lock`.
- Kiểm tra dữ liệu/evaluator yêu cầu Unicode runtime đúng protocol; không thay điều kiện này bằng runtime reporting.
- Tạo hình dùng môi trường riêng, `matplotlib==3.10.6`; xem [hướng dẫn reporting](src/analysis/test02d/README.md).
- Các file renderer đã gắn với bộ hình hiện tại được giữ nguyên để bảo toàn khả năng kiểm tra nguồn.

```sh
# Dùng runtime dữ liệu phù hợp protocol
python3 -B -m unittest discover -s tests -p 'test_protocol.py' -v
python3 -B -m unittest discover -s tests -p 'test_analysis*.py' -v
python3 -B -m unittest discover -s tests -p 'test_test02b.py' -v
python3 -B -m unittest discover -s tests -p 'test_test02c*.py' -v

# Dùng môi trường reporting cho nhóm test có Matplotlib
python -B -m unittest discover -s tests -p 'test_test02d*.py' -v
```

Bài toán có tám nhãn phát hiện khía cạnh: STAYINGPOWER, TEXTURE, SMELL, PRICE, OTHERS, COLOUR, SHIPPING, PACKING. Bảy khía cạnh cốt lõi có cảm xúc Positive, Neutral hoặc Negative; OTHERS được đánh giá riêng. Lịch sử huấn luyện neural vẫn có giới hạn `ASSUMED_MANIFEST_EQUIVALENCE`; refactor không thay đổi trạng thái này.

## Mã nguồn và công bố gốc

Các notebook `src/single_task_learning.ipynb`, `src/multi_task_learning.ipynb` và EDA được giữ từ baseline gốc. Chúng khác pipeline TF-IDF/BiGRU/PhoBERT hiện tại. Công trình PACLIC 2022 nghiên cứu các biến thể BiLSTM, BiGRU và Conv1D với single-task/multi-task learning, đồng thời cung cấp bộ dữ liệu được sử dụng.

[Tran, Le và Do — PACLIC 2022](https://aclanthology.org/2022.paclic-1.84/)

```bibtex
@inproceedings{tran-etal-2022-aspect,
  title = "Aspect-based Sentiment Analysis for {V}ietnamese Reviews about Beauty Product on {E}-commerce Websites",
  author = "Tran, Quang-Linh and Le, Phan Thanh Dat and Do, Trong-Hop",
  booktitle = "Proceedings of the 36th Pacific Asia Conference on Language, Information and Computation",
  month = oct,
  year = "2022",
  address = "Manila, Philippines",
  publisher = "De La Salle University",
  url = "https://aclanthology.org/2022.paclic-1.84",
  pages = "767--776"
}
```
