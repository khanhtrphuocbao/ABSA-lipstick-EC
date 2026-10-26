# Hướng dẫn báo cáo: Macro-F1 ceiling theo protocol v1

Đây là bổ sung cách diễn giải; không đổi công thức, evaluator, zero_division, targets hoặc freeze-lock v1.

Với K=21 labels cố định và m labels có gold support>0, mỗi label absent có F1=0 theo convention v1 
(kể cả khi có false positives). Vì vậy trên subset không rỗng: **Macro-F1 ≤ m/21**.
Đây là support-only upper bound đạt được bởi oracle sao chép gold, không phải dự đoán về mức model thực tế đạt được.

| Tập | Reviews | Supported/21 | Upper bound | Zero-support labels |
|---|---:|---:|---:|---|
| train | 12981 | 21/21 | 100.0000% | — |
| val | 1623 | 20/21 | 95.2381% | price:negative |
| original_test | 1623 | 21/21 | 100.0000% | — |
| test_nonoverlap_l1 | 1592 | 20/21 | 95.2381% | price:neutral |
| test_nonoverlap_l2 | 1585 | 20/21 | 95.2381% | price:neutral |
| test_placeholder_only | 8 | 0/21 | 0.0000% | colour:negative, colour:neutral, colour:positive, packing:negative, packing:neutral, packing:positive, price:negative, price:neutral, price:positive, shipping:negative, shipping:neutral, shipping:positive, smell:negative, smell:neutral, smell:positive, stayingpower:negative, stayingpower:neutral, stayingpower:positive, texture:negative, texture:neutral, texture:positive |
| test_without_placeholder | 1615 | 21/21 | 100.0000% | — |

## Cách diễn giải trong báo cáo thực nghiệm

- Validation thiếu PRICE-Negative: ngay cả oracle cũng chỉ có primary joint Macro-F1 20/21 ≈95,2381%. Đó là convention support, không phải lỗi model trên lớp chưa được kiểm tra.
- Non-overlap L1/L2 thiếu PRICE-Neutral: cùng ceiling 20/21. Không so raw Macro-F1 giữa các subsets như chúng có cùng độ khó hoặc composition.
- Placeholder-only test có 0/21 supported core labels: ceiling=0, core recall N/A; vẫn report false positives và OTHERS detection. Điểm 0 ở đây không chứng minh model không hiểu review.
- Conditional summary fixed7×fixed3 có support ceiling tương ứng m/21, vì mỗi aspect nhận trọng số bằng nhau; per-aspect conditional ceiling là số sentiment có support /3.
- Luôn report primary fixed21, observed-support macro với tên riêng, TP/FP/FN/support, Negative P/R và zero-support list. Không chia primary cho ceiling, không thay 21 bằng m trong primary.
- Support ceiling khác với best achievable score do annotation conflicts, model capacity hoặc input information; nó chỉ là giới hạn theo công thức và support.
- Chọn configuration bằng mean validation primary trên 3 seeds đã khóa; ceiling validation cố định cho mọi candidate nên không đổi model ranking hoặc tie-break.
- Nếu thay denominator, bỏ absent labels khỏi primary hoặc đổi zero_division: tạo protocol v2. Thêm bảng ceiling/cách diễn giải giữ nguyên protocol v1.

Nguồn kiểm chứng convention: [scikit-learn F1 docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
Các giá trị trên tính từ manifest/source đã freeze, không có model predictions.
