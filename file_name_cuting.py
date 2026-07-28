import os
def trim_file_names(dir_path, target_str, keep_target=True):
    """지정한 폴더 및 하위 폴더 내 모든 파일의 특정 문자열 뒤 이름을 제거합니다.

    Args:
        dir_path (str): 파일들이 있는 디렉토리 경로
        target_str (str): 제거할 기준 문자열
        keep_target (bool): 기준 문자열을 유지할지 여부
    """
    if not os.path.exists(dir_path):
        print(f"Error: The directory '{dir_path}' does not exist.")
        return

    renamed_count = 0

    print(f"파일 이름 변경 미리보기")
    changes = []

    # os.walk를 사용하여 모든 하위 디렉토리를 순회
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            name, ext = os.path.splitext(file_name)

            if target_str in name:
                idx = name.find(target_str)
                if keep_target:
                    new_name = name[: idx +len(target_str)] + ext
                else:
                    new_name = name[: idx] + ext

                if file_name != new_name:
                    new_path = os.path.join(root, new_name)
                    changes.append((file_path, new_path, file_name, new_name))
                    print(f"{file_name} -> {new_name}")

    if not changes:
        print("변경할 파일이 없습니다.")
        return

    confirm = input("위 변경 사항을 적용하시겠습니까? (y/n): ").strip().lower()
    if confirm == 'y':
        for old_path, new_path, old_name, new_name in changes:
            try:
                os.rename(old_path, new_path)
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {old_name} to {new_name}: {e}")
        print(f"{renamed_count}개의 파일 이름이 변경되었습니다.")
    else:
        print("파일 이름 변경이 취소되었습니다.")

if __name__ == "__main__":
    # 예시 사용법
    directory_path = "C:\\Users\\LGESLocalAdmin\\Desktop\\NFF_valid\\IMG_DATA\\TRAIN_TEST\\TEST\\CRATER_2"  # 파일들이 있는 디렉토리 경로
    target_string = "_CRACK_2"  # 제거할 기준 문자열
    trim_file_names(directory_path, target_string, keep_target=True)