"""
4번 세그멘테이션용 데이터셋 캡처 도구
- 매 REMIND_EVERY장마다 "시편을 움직이세요" 큰 알림을 띄워서 근접 중복 프레임 방지
- 기존 이미지가 있으면 이어서 번호 매김 (덮어쓰지 않음)
- 저장 경로: dataset/images/{specimen}_{순번}.png
- 사용법: 스페이스바=캡처, n=다음 시편으로 이름 변경, q=종료
"""

import os
import re
import glob
import cv2
import numpy as np
import pyrealsense2 as rs

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(_ROOT, "dataset", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
REMIND_EVERY = 3  # 이 장수마다 "시편 움직이기" 알림 (근접 중복 프레임 방지)
REMIND_FRAMES = 25  # 알림을 몇 프레임 동안 화면에 유지할지 (약 1초, 30fps 기준)


def existing_counts():
    """이미 저장된 specimenN_XXX.png를 스캔해서 {specimen_idx: 다음 shot 번호} 반환."""
    counts = {}
    for path in glob.glob(os.path.join(IMAGES_DIR, "specimen*_*.png")):
        m = re.match(r"specimen(\d+)_(\d+)\.png", os.path.basename(path))
        if m:
            spec, shot = int(m.group(1)), int(m.group(2))
            counts[spec] = max(counts.get(spec, 0), shot)
    return counts


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)

    counts = existing_counts()
    specimen_idx = max(counts.keys()) if counts else 1
    shot_idx = counts.get(specimen_idx, 0)
    since_last_move_reminder = 0
    remind_timer = 0

    print("SPACE=캡처, n=다음 시편, q=종료")
    print(f"이어서 시작: specimen{specimen_idx}, 다음 shot={shot_idx + 1}")
    print(f"{REMIND_EVERY}장마다 '시편을 움직이세요' 알림이 뜹니다 — 근접 중복 프레임 방지용")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())

            preview = img.copy()
            cv2.putText(preview, f"specimen{specimen_idx} shot{shot_idx}  SPACE=capture n=next-specimen q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if remind_timer > 0:
                cv2.rectangle(preview, (0, 0), (preview.shape[1], preview.shape[0]), (0, 0, 255), 12)
                cv2.putText(preview, "MOVE THE SPECIMEN (시편을 움직이세요)",
                            (40, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                remind_timer -= 1

            cv2.imshow("capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                shot_idx += 1
                fname = f"specimen{specimen_idx}_{shot_idx:03d}.png"
                cv2.imwrite(os.path.join(IMAGES_DIR, fname), img)
                print(f"저장: {fname}")

                since_last_move_reminder += 1
                if since_last_move_reminder >= REMIND_EVERY:
                    since_last_move_reminder = 0
                    remind_timer = REMIND_FRAMES
            elif key == ord('n'):
                specimen_idx += 1
                shot_idx = counts.get(specimen_idx, 0)
                since_last_move_reminder = 0
                print(f"다음 시편으로 전환: specimen{specimen_idx} (다음 shot={shot_idx + 1})")
            elif key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
