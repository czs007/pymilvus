# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from minio import Minio
from minio.error import S3Error

from pymilvus.orm.schema import CollectionSchema

from .constants import (
    DEFAULT_BUCKET_NAME,
    BulkFileType,
    JSON_SUFFIX, NUMPY_SUFFIX
)

from .bulk_writer import BulkWriter

logger = logging.getLogger("remote_bulk_writer")
logger.setLevel(logging.DEBUG)


class RemoteBulkWriter(BulkWriter):
    class ConnectParam:
        def __init__(
                self,
                bucket_name: str = DEFAULT_BUCKET_NAME,
                endpoint: Optional[str] = None,
                access_key: Optional[str] = None,
                secret_key: Optional[str] = None,
                secure: bool = False,
                session_token: Optional[str] = None,
                region: Optional[str] = None,
                http_client: Any = None,
                credentials: Any = None,
        ):
            self._bucket_name = bucket_name
            self._endpoint = endpoint
            self._access_key = access_key
            self._secret_key = secret_key
            self._secure = (secure,)
            self._session_token = (session_token,)
            self._region = (region,)
            self._http_client = (http_client,)  # urllib3.poolmanager.PoolManager
            self._credentials = (credentials,)  # minio.credentials.Provider

    def __init__(
            self,
            connect_param: ConnectParam,
            schema: CollectionSchema,
            data_path: str,
            file_type: BulkFileType = BulkFileType.NPY,
            **kwargs,
    ):
        super().__init__(schema, str(data_path), file_type, **kwargs)
        self._connect_param = connect_param
        self._client = None
        self._get_client()
        logger.info(f"Remote buffer writer initialized, target path: {self._data_path}")

    def _get_client(self):
        try:
            if self._client is None:

                def arg_parse(arg: Any):
                    return arg[0] if isinstance(arg, tuple) else arg

                self._client = Minio(
                    endpoint=arg_parse(self._connect_param._endpoint),
                    access_key=arg_parse(self._connect_param._access_key),
                    secret_key=arg_parse(self._connect_param._secret_key),
                    secure=arg_parse(self._connect_param._secure),
                    session_token=arg_parse(self._connect_param._session_token),
                    region=arg_parse(self._connect_param._region),
                    http_client=arg_parse(self._connect_param._http_client),
                    credentials=arg_parse(self._connect_param._credentials),
                )
            else:
                return self._client
        except Exception as err:
            logger.error(f"Failed to connect MinIO/S3, error: {err}")
            raise

    def commit(self, **kwargs):
        super().commit(target=self._upload, **kwargs)  # reset the buffer size

    def _remote_exists(self, file: str) -> bool:
        try:
            minio_client = self._get_client()
            minio_client.stat_object(bucket_name=self._connect_param._bucket_name, object_name=file)
        except S3Error as err:
            if err.code == "NoSuchKey":
                return False
            self._throw(f"Failed to stat MinIO/S3 object, error: {err}")
        return True

    def _upload(self, call_back: Optional[Callable] = None):
        self._working_thread[threading.current_thread().name] = threading.current_thread()
        target_path = self._data_path
        if self._flush_count:
            target_path = Path.joinpath(self._data_path, str(self._flush_count))

        self._flush_count = self._flush_count + 1

        old_buffer = super()._new_buffer()
        file_list = old_buffer.persist_file_obj()
        suffix = ".json"
        if old_buffer.file_type == BulkFileType.NPY:
            suffix = ".npy"
        else:
            file_list = [target_path, file_list[1]]

        remote_files = []
        try:
            logger.info("Prepare to upload files")
            minio_client = self._get_client()
            found = minio_client.bucket_exists(self._connect_param._bucket_name)
            if not found:
                self._throw(f"MinIO bucket '{self._connect_param._bucket_name}' doesn't exist")

            for file_path, file in file_list:
                minio_file_path = file_path + suffix
                if self._remote_exists(minio_file_path):
                    logger.info(
                        f"Remote file '{minio_file_path}' already exists, will overwrite it"
                    )

                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                minio_client.put_object(
                    self._connect_param._bucket_name,
                    minio_file_path,
                    file,
                    file_size,
                )
                logger.info(f"Upload file '{file_path}' to '{self.data_path}'")

                remote_files.append(str(minio_file_path))
        except Exception as e:
            self._throw(f"Failed to call MinIO/S3 api, error: {e}")

        logger.info(f"Successfully upload files: {file_list}")
        self._output_files.extend(remote_files)
        if call_back:
            call_back(file_list)
        return remote_files
