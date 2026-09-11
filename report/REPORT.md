# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
```
  {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }
```
- Record này mô tả toàn ảnh như thế nào?

Model told the whole image is cab

- Ai định nghĩa class list mà checkpoint có thể dự đoán?

Dataset Creator

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

taxonomy is defined, id help detect what item in taxonomy, class name for help human understand.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

1. Guideline should force return 1 class in the taxonomy. 
2. Guideline can choose 1 class and avoid other classes.

- Vì sao model score không phải ground truth?

1. score is model produced, that's not ground truth. Ground truth should be defined in dataset use to be train model instead of result of model prediction.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 5,
    "class_name": "bus",
    "score": 0.912558,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
  }

- Diễn giải vị trí box bằng lời:

The size of box is: bbox_width and bbox_height
The top-left position is: x=93.17,y=187.95 and bottom-right is: x=223.01,y=320.91

- So sánh số prediction ở hai threshold:

Score is different, first record score is 0.912558, second record score is 0.863075.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

Compare with classification, need see more details, that's mean it will take time.

- Đề xuất một quy tắc box chặt:

Increase score_threshold

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

1. Increase labels: We can avoid it or should has label good than only the whole class, example we can has a label is `hand_of_human` beside the tag `human` if in the image only has hand. 
2. Strict guideline, need the whole object instead of a part of object.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

```
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
    ]
```

- Polygon bổ sung chi tiết gì so với box?

instance_id, polygon_point_count, polygon_xy.

- `instance_id` dùng để làm gì và không phải loại ID nào?

instance_id detected a polygon.

- Đề xuất một quy tắc biên mask:

1. Defined a polygon_point_count
2. Space between polygon_xy_point should have a rule example, minimum distance is 2mm, maximum distance is 5mm.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

1. Guideline should has a section or additional docs for edge case.
2. If edge case not cover, raise an escalation or set a class is need double check.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | x | x | define class | double check |
| Phát hiện vật thể | x | x | object | double check |
| Instance segmentation | x | x | polygon | double check |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:

1. Not share on public.
2. Do not install package unauthorize.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

Annotator Lead

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
