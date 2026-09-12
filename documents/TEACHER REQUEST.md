Ngoài ra, Thầy đã rà soát kỹ bản thảo “Robust Face Presentation Attack Detection for eKYC Under Image Quality Degradations”. Hướng nghiên cứu phù hợp để phát triển thành paper hội thảo: bài có pipeline tương đối rõ, subject-disjoint split, fixed test set và kết quả cho thấy quality-aware augmentation cải thiện đáng kể robustness dưới các degradation. Tuy nhiên, bản hiện tại chưa nên xem là final vì còn một số vấn đề về protocol, cách diễn giải kết quả và độ mạnh của evidence. Đặc biệt, các con số trong bảng nhìn chung khá nhất quán, nhưng câu chuyện khoa học đang mạnh hơn những gì thiết kế thực nghiệm thực sự chứng minh.
Nhóm tập trung sửa các điểm sau, theo thứ tự ưu tiên:

1. Vấn đề quan trọng nhất: kết quả hiện tại mới dựa trên một seed duy nhất nhưng claim robustness đang khá mạnh.
   Toàn bộ experiment hiện sử dụng seed 123, hai model cũng chỉ có một training run và chọn checkpoint epoch 18. Trong khi đó bài kết luận augmentation làm mean F1 tăng từ 0.9226 lên 0.9743, mean ACER giảm từ 0.1153 xuống 0.0390 và worst-case cũng cải thiện rất mạnh.
   Những khác biệt này lớn và đáng chú ý, nhưng một seed chưa đủ để chứng minh training strategy ổn định.
   Nhóm nên ưu tiên chạy tối thiểu 3 seeds, tốt hơn 5 seeds, cho E01 và E07 rồi báo cáo mean ± SD cho các metric chính. Nếu chi phí GPU không cho phép chạy lại toàn bộ 16 degradation × nhiều seeds thì ít nhất chạy nhiều seeds cho clean + một số degradation đại diện/severe.
   Nếu không chạy thêm được, phải hạ claim thành evidence from a controlled single-seed experiment, đồng thời đưa single-seed evaluation vào Limitations.
2. Cách dùng threshold = 0.5 làm APCER/BPCER/ACER chưa đủ thuyết phục.
   Paper cố định threshold 0.5 cho toàn bộ test và chính Limitations cũng thừa nhận không có threshold tuning trên validation.
   Điểm này quan trọng vì F1, APCER, BPCER và ACER đều phụ thuộc threshold. Vì vậy một phần improvement có thể liên quan đến calibration/score distribution, không chỉ discrimination robustness.
   Nhóm nên chọn threshold hoàn toàn trên validation set, sau đó freeze threshold trước khi chạy clean test và degradation test. Không được chọn threshold trên test.
   Nếu giữ threshold 0.5 thì phải nói rõ đây là fixed-threshold protocol, không hàm ý 0.5 là operating point tối ưu cho eKYC.
3. Có một điểm phương pháp rất hay nhưng bài chưa khai thác: training degradation và testing degradation không cùng severity range.
   Training augmentation dùng:
   • JPEG: 50–90, nhưng test xuống JPEG 30;
   • Resize: 0.5–1.0, nhưng test xuống 0.25;
   • Blur sigma: 0.5–2.0, nhưng test lên 3.0;
   • Brightness: 0.7–1.3, nhưng test dùng cả 0.6 và 1.4.
   Trong khi Table III đưa các mức degradation mạnh hơn vào test.
   Đây không phải lỗi, ngược lại có thể trở thành một điểm mạnh. Nhưng nhóm phải phân biệt:
   in-range degradation và out-of-range/severity extrapolation.
   Nếu E07 vẫn cải thiện rõ ở JPEG 30, resize 25%, blur sigma 3.0... thì evidence sẽ thú vị hơn nhiều: model không chỉ học đúng mức augmentation đã thấy mà còn có khả năng chịu degradation nghiêm trọng hơn training range.
4. Claim “clean performance essentially preserved” cần diễn đạt chính xác hơn.
   Clean performance thực tế đều giảm:
   F1: 0.9868 → 0.9831
   ACER: 0.0211 → 0.0256
   ROC-AUC: 0.9978 → 0.9969.
   Mức giảm nhỏ nên câu chuyện chung vẫn hợp lý, nhưng chưa có confidence interval/statistical test để kết luận hai model “essentially equal”.
   Nên viết thận trọng hơn:
   robust training incurs a small clean-performance cost while substantially improving degradation robustness.
   Cách này chính xác hơn và tạo ra một clean-performance–robustness trade-off rất tự nhiên cho bài.
5. Chưa có statistical significance/uncertainty analysis.
   Test có tới 40,643 ảnh nên nhóm có điều kiện thực hiện phân tích uncertainty khá tốt.
   Hiện paper chỉ báo một giá trị cho từng metric. Reviewer hoàn toàn có thể hỏi liệu chênh lệch E01–E07 có đáng tin cậy hay chỉ do sampling/training randomness.
   Nếu chưa đủ thời gian chạy nhiều seeds, có thể bổ sung paired bootstrap confidence intervals trên fixed test set cho các chênh lệch chính. Tuy nhiên bootstrap test samples không thay thế được multi-seed analysis về training variability; cần nói rõ hai vấn đề này khác nhau.
6. “Subject-disjoint” là điểm rất quan trọng nhưng cách xác định subject hiện chưa đủ chắc chắn.
   Paper viết subject identity được “derived from image filename prefixes”.
   Nhóm phải kiểm tra lại metadata của CelebA-Spoof/mirror để chứng minh prefix thực sự tương ứng với subject identity, chứ không chỉ là folder/image identifier.
   Cần báo cáo ít nhất:
   • số subjects trong train;
   • số subjects validation;
   • số subjects test;
   • intersection train–val = 0;
   • intersection train–test = 0;
   • intersection val–test = 0.
   Nếu không xác nhận được điều này thì claim “subject-disjoint” là một methodological risk lớn.
7. Robustness hiện mới là synthetic corruption robustness, chưa phải robustness của eKYC thực tế.
   16 conditions đều được tạo bằng degradation nhân tạo: JPEG, resize, blur, noise và brightness.
   Bài đã ghi limitation rằng chưa có cross-dataset/device-level evaluation và degradations là synthetic, đây là đúng. Tuy nhiên Abstract, Introduction và Conclusion vẫn cần giữ đúng phạm vi này.
   Không nên mở rộng thành:
   “robust for real-world eKYC deployment”.
   Nên giới hạn thành: “robustness under controlled image-quality degradations relevant to eKYC capture conditions”.
8. “Lightweight” mới được chứng minh bằng parameter count/model size, chưa được chứng minh ở deployment efficiency.
   Paper báo MobileNetV2 có 2,225,153 parameters và 8.62 MB, nhưng lại thừa nhận chưa benchmark latency.
   Do đó có thể gọi MobileNetV2 là lightweight architecture, nhưng chưa nên suy rộng thành real-time/mobile/eKYC-ready.
   Nếu còn thời gian, chỉ cần thêm một experiment nhỏ:
   CPU inference latency, images/s, model size và peak memory.
   Không cần thêm model mới.
9. Related Work hiện quá mỏng so với câu chuyện robustness.
   Related Work hiện chủ yếu dựa trên texture PAD, MobileNetV2, generic corruption robustness và augmentation.
   Nhưng paper lại muốn contribution nằm ở Face PAD robustness under quality degradation. Nhóm cần bổ sung trực tiếp các nghiên cứu gần đây về:
   Face PAD robustness → domain generalization → cross-dataset PAD → image-quality degradation → mobile/eKYC PAD.
   Sau đó mới chỉ ra gap:
   các nghiên cứu trước đạt clean/cross-domain PAD performance, nhưng empirical evidence về lightweight PAD dưới nhiều mức image-quality degradation và quality-aware robustness training còn hạn chế.
   Không nên claim “first” nếu chưa có systematic literature evidence.
10. Table VI đang quá lớn và khó đọc, không phù hợp với paper hội thảo.
    Table VI chứa toàn bộ 16 conditions × rất nhiều metrics nên chiếm diện tích lớn và chữ rất nhỏ ở trang 6–7. Trong khi Fig. 4 đã trực quan hóa F1/ACER, Table VII lại tổng hợp mean và Table IX tiếp tục tổng hợp theo degradation category. Có sự lặp thông tin khá nhiều.
    Thầy đề nghị:
    • Table VI chỉ giữ F1, ROC-AUC, APCER, BPCER, ACER;
    • bỏ các cột delta có thể suy ra trực tiếp;
    • chuyển full results sang supplementary/repository nếu hội thảo cho phép;
    • Fig. 4 giữ lại vì biểu diễn xu hướng degradation tốt hơn bảng.
11. Table X gồm loss của toàn bộ 20 epochs cũng không cần thiết.
    Bảng này chiếm gần một trang nhưng Fig. 6 đã biểu diễn training/validation curves.
    Bỏ Table X, giữ Fig. 6.
    Trong text chỉ cần ghi:
    “Both models reached their minimum validation loss at epoch 18.”
    Việc này giúp tiết kiệm gần một trang mà không làm mất nội dung khoa học.

12. Cần làm rõ model initialization.
    Methodology mô tả MobileNetV2, ImageNet normalization, optimizer, learning rate... khá đầy đủ nhưng chưa thấy nói rõ MobileNetV2 được:
    trained from scratch hay initialized with ImageNet-pretrained weights.
    Đây là thông tin bắt buộc để reproducibility. Nếu pretrained thì phải ghi rõ weights/version và phần classifier nào được thay thế.
    Hướng nhóm sửa ở vòng tiếp theo
    Thầy không đề nghị bổ sung thêm nhiều architecture. Điểm mạnh tiềm năng của paper không phải “MobileNetV2 tốt hơn model khác”, mà là câu chuyện:
    clean PAD → controlled quality degradation → performance collapse → quality-aware augmentation → robustness recovery → small clean-performance cost.
    Nhóm làm theo thứ tự:
    (1) xác minh subject-disjoint split →
    (2) kiểm tra threshold protocol →
    (3) bổ sung multi-seed hoặc uncertainty analysis →
    (4) phân biệt in-range/out-of-range degradation →
    (5) hạ các claim real-world/eKYC cho đúng evidence →
    (6) bổ sung Related Work trực tiếp về robust PAD →
    (7) rút Table VI và bỏ Table X →
    (8) cuối cùng mới đồng bộ Abstract – Contributions – Results – Discussion – Conclusion.
    Nếu làm được các điểm trên, bài có hướng khá tốt để phát triển thành bài hội thảo, vì kết quả quan trọng hiện đã xuất hiện khá rõ: robust model giảm mean ACER từ 0.1153 xuống 0.0390 và cải thiện worst-case ACER từ 0.3310 xuống 0.1012. Điều nhóm cần lúc này không phải làm kết quả “đẹp hơn”, mà là làm cho protocol đủ chặt và claim đúng với evidence.
