# TH3-Gaussian-Informer
🌦️ Dự báo Nhiệt độ và Độ ẩm tại Hà Nội bằng Gaussian Process & Informer
Thời gian dữ liệu: 12/05/2024 – 12/05/2025
Ngôn ngữ: Python

📌 Mô tả dự án
Dự án này nhằm so sánh hai phương pháp tiên tiến trong dự báo chuỗi thời gian:

Gaussian Process (GP) – một phương pháp thống kê Bayesian không tham số

Informer – một biến thể tối ưu của Transformer cho chuỗi thời gian dài

Cả hai được áp dụng để dự báo nhiệt độ và độ ẩm tại Hà Nội, dựa trên dữ liệu thời tiết thực tế thu thập từ trang RP5.ru.

📊 Dữ liệu sử dụng
Nguồn: https://rp5.ru

Địa điểm: Hà Nội (airport)

Khoảng thời gian: 12/05/2024 – 12/05/2025

Đặc trưng chính:

T: Nhiệt độ (°C)

U: Độ ẩm tương đối (%)

P, P0: Áp suất khí quyển (hPa)

Ff: Tốc độ gió (m/s)

Td: Điểm sương (°C)

VV: Tầm nhìn xa (km)

⚙️ Tiền xử lý dữ liệu
Loại bỏ các cột không cần thiết (c, DD, WW, W'W', ff10, Unnamed: 13)

Xử lý thiếu bằng nội suy tuyến tính

Chuyển đổi thời gian về định dạng chuẩn datetime

Chuẩn hóa các biến số bằng StandardScaler

Trích xuất đặc trưng thời gian: giờ, ngày, tháng, ngày trong tuần

🧠 Mô hình
✅ Gaussian Process (GP)
Kernel sử dụng: ConstantKernel * RBF

Tối ưu siêu tham số: n_restarts_optimizer=10

Dữ liệu huấn luyện: 1000 bản ghi đầu tiên (80% train / 20% test)

Đầu ra: Dự báo T và U kèm theo độ lệch chuẩn

✅ Informer
Kiến trúc: Encoder-Decoder (theo Transformer)

Thông số chính:

seq_len: 96 | label_len: 48 | pred_len: 24

d_model: 512 | n_heads: 8

e_layers: 2 | d_layers: 1

dropout: 0.05 | attention: "prob"

Tối ưu hóa: Adam, LR = 0.001, batch size = 32, epochs = 10

📈 Kết quả
Mô hình	Biến	MAE	RMSE
Gaussian Process	T	0.3289	0.433
Informer	T	(đang cập nhật)	(đang cập nhật)

GP cho kết quả tốt hơn trên dữ liệu nhỏ và ngắn hạn.
Informer mạnh hơn trong mô hình hóa chuỗi dài và nhiều đặc trưng.

📌 So sánh & Nhận xét
Tiêu chí	Gaussian Process	Informer
Độ chính xác ngắn hạn	✅ Tốt	Trung bình
Dự báo dài hạn	❌ Kém	✅ Mạnh
Độ không chắc chắn	✅ Có thể lượng hóa	❌ Không có
Hiệu suất trên dữ liệu lớn	❌ Kém (O(n³))	✅ Tốt
Đòi hỏi dữ liệu lớn	❌ Không	✅ Cần dữ liệu nhiều
Tùy chỉnh siêu tham số	Dễ	Phức tạp

🚀 Hướng phát triển
GP:

Áp dụng Sparse GP

Dùng các kernel phức tạp hơn: Periodic, Spectral Mixture

Informer:

Kết hợp Bayesian để định lượng độ không chắc chắn

Dùng Transfer Learning, AutoML cho chọn siêu tham số

Kết hợp:

Ensemble GP + Informer

Dự báo dài hạn với Informer, định lượng rủi ro với GP
