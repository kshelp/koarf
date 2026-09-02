from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

import requests


class KISTIQAClient:
    """
    KISTI AIDA 국내 논문 QA 데이터셋 다운로드 클라이언트.

    KISTI AIDA Open API Token을 환경변수 KISTI_AIDA_TOKEN으로
    설정하여 사용하는 것을 권장한다.
    """

    DEFAULT_DATASET_ID = "21b21974-6efd-4581-b9df-699f91f5bc98"

    def __init__(
        self,
        token: Optional[str] = None,
        dataset_id: str = DEFAULT_DATASET_ID,
        base_url: str = "https://aida.kisti.re.kr",
    ):
        self.token = token or os.getenv("KISTI_AIDA_TOKEN")

        if not self.token:
            raise ValueError(
                "KISTI_AIDA_TOKEN 환경변수를 설정하거나 "
                "token을 직접 전달하세요."
            )

        self.dataset_id = dataset_id
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def get_dataset_info(self) -> dict:
        """
        KISTI AIDA 데이터셋 정보를 조회한다.
        """

        url = f"{self.base_url}/api/data/{self.dataset_id}"

        response = requests.get(
            url,
            headers=self._headers(),
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    def download(
        self,
        output_dir: str = "./data",
        filename: str = "kisti_korean_paper_qa.zip",
        chunk_size: int = 1024 * 1024,
        extract: bool = False,
    ) -> Path:
        """
        데이터셋 ZIP 파일을 다운로드한다.

        Parameters
        ----------
        output_dir:
            다운로드 디렉터리
        filename:
            저장할 파일명
        chunk_size:
            다운로드 chunk 크기
        extract:
            True이면 다운로드 후 자동 압축 해제
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        zip_path = output_path / filename

        url = self._get_download_url()

        print(f"Download URL: {url}")
        print(f"Output: {zip_path}")

        with requests.get(
            url,
            headers=self._headers(),
            stream=True,
            timeout=60,
        ) as response:

            response.raise_for_status()

            total_size = int(
                response.headers.get("content-length", 0)
            )

            downloaded = 0

            with open(zip_path, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=chunk_size
                ):

                    if not chunk:
                        continue

                    f.write(chunk)

                    downloaded += len(chunk)

                    if total_size:
                        percent = downloaded / total_size * 100

                        print(
                            f"\rProgress: "
                            f"{percent:6.2f}% "
                            f"({downloaded / 1024**2:.1f} MB)",
                            end="",
                        )

        print()

        if extract:
            self.extract(zip_path)

        return zip_path

    def _get_download_url(self) -> str:
        """
        AIDA API에서 실제 파일 다운로드 URL을 조회한다.

        주의:
        KISTI AIDA API의 실제 endpoint가 변경될 수 있으므로
        API 응답 구조에 따라 수정할 수 있도록 별도 메서드로 분리한다.
        """

        url = (
            f"{self.base_url}/api/data/"
            f"{self.dataset_id}/files"
        )

        response = requests.get(
            url,
            headers=self._headers(),
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        # 일반적인 API 응답 형태를 순차적으로 확인
        if isinstance(data, dict):

            if "download_url" in data:
                return data["download_url"]

            if "url" in data:
                return data["url"]

            if "files" in data:
                files = data["files"]

                for file in files:

                    name = file.get("name", "")

                    if "국내_논문_QA_데이터셋" in name:
                        return (
                            file.get("download_url")
                            or file.get("url")
                        )

        raise RuntimeError(
            "KISTI AIDA API 응답에서 "
            "다운로드 URL을 찾을 수 없습니다. "
            f"응답: {data}"
        )

    @staticmethod
    def extract(
        zip_path: str | Path,
        output_dir: Optional[str] = None,
    ) -> Path:

        zip_path = Path(zip_path)

        if output_dir is None:
            output_dir = zip_path.parent / zip_path.stem

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting: {zip_path}")
        print(f"Destination: {output_dir}")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)

        print("Extraction completed.")

        return output_dir