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

import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from pymilvus.client.types import (
    DataType,
)
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema

from .constants import (
    DYNAMIC_FIELD_NAME,
    NUMPY_TYPE_CREATOR,
    BulkFileType,
    MB
)

logger = logging.getLogger("bulk_buffer")
logger.setLevel(logging.DEBUG)

MAX_SPOOL_FILE_SIZE = 512 * MB


class Buffer:
    def __init__(
            self,
            schema: CollectionSchema,
            file_type: BulkFileType = BulkFileType.NPY,
    ):
        self._buffer = {}
        self._fields = {}
        self._file_type = file_type
        for field in schema.fields:
            if field.is_primary and field.auto_id:
                continue
            self._buffer[field.name] = []
            self._fields[field.name] = field

        if len(self._buffer) == 0:
            self._throw("Illegal collection schema: fields list is empty")

        # dynamic field, internal name is '$meta'
        if schema.enable_dynamic_field:
            self._buffer[DYNAMIC_FIELD_NAME] = []

    @property
    def row_count(self) -> int:
        if len(self._buffer) == 0:
            return 0

        for k in self._buffer:
            return len(self._buffer[k])
        return None

    def _throw(self, msg: str):
        logger.error(msg)
        raise MilvusException(message=msg)

    def append_row(self, row: dict):
        dynamic_values = {}
        if DYNAMIC_FIELD_NAME in row and not isinstance(row[DYNAMIC_FIELD_NAME], dict):
            self._throw(f"Dynamic field '{DYNAMIC_FIELD_NAME}' value should be JSON format")

        for k in row:
            if k == DYNAMIC_FIELD_NAME:
                dynamic_values.update(row[k])
                continue

            if k not in self._buffer:
                dynamic_values[k] = row[k]
            else:
                self._buffer[k].append(row[k])

        if DYNAMIC_FIELD_NAME in self._buffer:
            self._buffer[DYNAMIC_FIELD_NAME].append(json.dumps(dynamic_values))

    @property
    def file_type(self):
        return self._file_type

    def persist_file_obj(self) -> list(tuple):
        self._verify()

        # output files
        if self._file_type == BulkFileType.NPY:
            return self._persist_npy_file_obj()
        if self._file_type == BulkFileType.JSON_RB:
            return self._persist_json_file_obj()

        self._throw(f"Unsupported file tpye: {self._file_type}")
        return []

    def _persist_npy_file_obj(self):
        file_list = []
        ret = []
        for k in self._buffer:
            file_name = k
            file_list.append(file_name)
            try:
                dt = None
                field_schema = self._fields[k]
                if field_schema.dtype.name in NUMPY_TYPE_CREATOR:
                    dt = NUMPY_TYPE_CREATOR[field_schema.dtype.name]

                # for JSON field, convert to string array
                if field_schema.dtype == DataType.JSON:
                    str_arr = []
                    for val in self._buffer[k]:
                        str_arr.append(json.dumps(val))
                    self._buffer[k] = str_arr

                arr = np.array(self._buffer[k], dtype=dt)
                tmp = tempfile.SpooledTemporaryFile(max_size=MAX_SPOOL_FILE_SIZE)
                np.save(tmp, arr)
                tmp.seek(0)
                ret.append(file_name, tmp)
            except Exception as e:
                self._throw(f"Failed to persist SpooledTemporaryFile (column-based) {file_name}, error: {e}")

            logger.info(f"Successfully persist SpooledTemporaryFile (column-based) {file_name}")

        if len(file_list) != len(self._buffer):
            logger.error("Some of fields were not persisted successfully, abort the files")
            file_list.clear()
            self._throw("Some of fields were not persisted successfully, abort the files")

        return ret

    def _persist_json_file_obj(self):
        rows = []
        row_count = len(next(iter(self._buffer.values())))
        row_index = 0
        while row_index < row_count:
            row = {}
            for k, v in self._buffer.items():
                row[k] = v[row_index]
            rows.append(row)
            row_index = row_index + 1

        data = {
            "rows": rows,
        }

        tmp = tempfile.SpooledTemporaryFile(max_size=MAX_SPOOL_FILE_SIZE)
        try:
            json.dump(data, tmp, indent=2)
        except Exception as e:
            self._throw(f"Failed to persist row-based json file, error: {e}")

        logger.info(f"Successfully persist row-based json file")
        tmp.seek(0)
        return [("", tmp)]

    def _verify(self):
        # verify row count of fields are equal
        row_count = -1
        for k in self._buffer:
            if row_count < 0:
                row_count = len(self._buffer[k])
            elif row_count != len(self._buffer[k]):
                self._throw(
                    "Column `{}` row count {} doesn't equal to the first column row count {}".format(
                        k, len(self._buffer[k]), row_count
                    )
                )

    def persist(self, local_path: str) -> list:
        self._verify()

        # output files
        if self._file_type == BulkFileType.NPY:
            return self._persist_npy(local_path)
        if self._file_type == BulkFileType.JSON_RB:
            return self._persist_json(local_path)

        self._throw(f"Unsupported file tpye: {self._file_type}")
        return []

    def _persist_npy(self, local_path: str):
        file_list = []
        for k in self._buffer:
            full_file_name = Path(local_path).joinpath(k + ".npy")
            file_list.append(str(full_file_name))
            try:
                Path(local_path).mkdir(exist_ok=True)

                # numpy data type specify
                dt = None
                field_schema = self._fields[k]
                if field_schema.dtype.name in NUMPY_TYPE_CREATOR:
                    dt = NUMPY_TYPE_CREATOR[field_schema.dtype.name]

                # for JSON field, convert to string array
                if field_schema.dtype == DataType.JSON:
                    str_arr = []
                    for val in self._buffer[k]:
                        str_arr.append(json.dumps(val))
                    self._buffer[k] = str_arr

                arr = np.array(self._buffer[k], dtype=dt)
                np.save(str(full_file_name), arr)
            except Exception as e:
                self._throw(f"Failed to persist column-based file {full_file_name}, error: {e}")

            logger.info(f"Successfully persist column-based file {full_file_name}")

        if len(file_list) != len(self._buffer):
            logger.error("Some of fields were not persisted successfully, abort the files")
            for f in file_list:
                Path(f).unlink()
            Path(local_path).rmdir()
            file_list.clear()
            self._throw("Some of fields were not persisted successfully, abort the files")

        return file_list

    def _persist_json(self, local_path: str):
        rows = []
        row_count = len(next(iter(self._buffer.values())))
        row_index = 0
        while row_index < row_count:
            row = {}
            for k, v in self._buffer.items():
                row[k] = v[row_index]
            rows.append(row)
            row_index = row_index + 1

        data = {
            "rows": rows,
        }
        file_path = Path(local_path + ".json")
        try:
            with file_path.open("w") as json_file:
                json.dump(data, json_file, indent=2)
        except Exception as e:
            self._throw(f"Failed to persist row-based file {file_path}, error: {e}")

        logger.info(f"Successfully persist row-based file {file_path}")
        return [str(file_path)]
