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
import time
from threading import Lock, Thread
from pathlib import Path

from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema

from .buffer import (
    Buffer,
)

from .constants import (
    MB,
    TYPE_SIZE,
    TYPE_VALIDATOR,
    BulkFileType,
)

logger = logging.getLogger("bulk_writer")
logger.setLevel(logging.DEBUG)


class BulkWriter:
    def __init__(
            self,
            schema: CollectionSchema,
            data_path: str,
            file_type: BulkFileType = BulkFileType.NPY,
            **kwargs,
    ):
        self._segment_size = kwargs.get("segment_size", 512 * MB)
        self._schema = schema
        self._buffer_size = 0
        self._data_path = data_path
        self._buffer_row_count = 0
        self._file_type = file_type
        self._output_files = []
        self._flush_count = 0
        self._working_thread = {}

        self._buffer_lock = Lock()

        if len(self._schema.fields) == 0:
            self._throw("collection schema fields list is empty")

        if self._schema.primary_field is None:
            self._throw("primary field is null")

        self._buffer = None
        self._new_buffer()
        self._target_func = None

    @property
    def data_path(self):
        return self._data_path

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def buffer_row_count(self):
        return self._buffer_row_count

    @property
    def segment_size(self):
        return self._segment_size

    @segment_size.setter
    def segment_size(self, size):
        self._segment_size = size

    def _new_buffer(self):
        old_buffer = self._buffer
        with self._buffer_lock:
            self._buffer = Buffer(self._schema, self._file_type)
        return old_buffer

    def append_row(self, row: dict, **kwargs):
        self._verify_row(row)
        with self._buffer_lock:
            self._buffer.append_row(row)

        if super().buffer_size > super().segment_size:
            self.commit(_async=True)

    def commit(self, **kwargs):
        while len(self._working_thread) > 0:
            logger.info("Previous flush action is not finished, waiting...")
            time.sleep(0.5)

        logger.info(
            f"Prepare to flush buffer, row_count: {super().buffer_row_count}, size: {super().buffer_size}"
        )
        _async = kwargs.get("_async", False)
        call_back = kwargs.get("call_back", None)
        target_func = kwargs.get("target", None)
        if callable(target_func):
            x = Thread(target=target_func, args=(call_back,))
            x.start()
            if not _async:
                logger.info("Wait flush to finish")
                x.join()
        self._buffer_size = 0
        self._buffer_row_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object):
        super().__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        self._exit()

    @property
    def output_files(self):
        return self._output_files

    def _throw(self, msg: str):
        logger.error(msg)
        raise MilvusException(message=msg)

    def _verify_row(self, row: dict):
        if not isinstance(row, dict):
            self._throw("The input row must be a dict object")

        row_size = 0
        for field in self._schema.fields:
            if field.is_primary and field.auto_id:
                if field.name in row:
                    self._throw(
                        f"The primary key field '{field.name}' is auto-id, no need to provide"
                    )
                else:
                    continue

            if field.name not in row:
                self._throw(f"The field '{field.name}' is missed in the row")

            dtype = DataType(field.dtype)
            validator = TYPE_VALIDATOR[dtype.name]
            if dtype in {DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR}:
                dim = field.params["dim"]
                if not validator(row[field.name], dim):
                    self._throw(
                        f"Illegal vector data for vector field: '{dtype.name}',"
                        f" dim is not {dim} or type mismatch"
                    )

                vec_size = (
                    len(row[field.name]) * 4
                    if dtype == DataType.FLOAT_VECTOR
                    else len(row[field.name]) / 8
                )
                row_size = row_size + vec_size
            elif dtype == DataType.VARCHAR:
                max_len = field.params["max_length"]
                if not validator(row[field.name], max_len):
                    self._throw(
                        f"Illegal varchar value for field '{dtype.name}',"
                        f" length exceeds {max_len} or type mismatch"
                    )

                row_size = row_size + len(row[field.name])
            elif dtype == DataType.JSON:
                if not validator(row[field.name]):
                    self._throw(f"Illegal varchar value for field '{dtype.name}', type mismatch")

                row_size = row_size + len(row[field.name])
            else:
                if not validator(row[field.name]):
                    self._throw(
                        f"Illegal scalar value for field '{dtype.name}', value overflow or type mismatch"
                    )

                row_size = row_size + TYPE_SIZE[dtype.name]

        self._buffer_size = self._buffer_size + row_size
        self._buffer_row_count = self._buffer_row_count + 1

    def _exit(self):
        if Path(self._data_path).exists() and not any(Path(self._data_path).iterdir()):
            Path(self._data_path).rmdir()
            logger.info(f"Delete empty directory '{self._data_path}'")

        if len(self._working_thread) > 0:
            for k, th in self._working_thread.items():
                logger.info(f"Wait flush thread '{k}' to finish")
                th.join()