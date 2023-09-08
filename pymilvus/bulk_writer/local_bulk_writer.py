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
import threading
from pathlib import Path
from typing import Callable, Optional

from .bulk_writer import BulkWriter

logger = logging.getLogger("local_bulk_writer")
logger.setLevel(logging.DEBUG)


class LocalBulkWriter(BulkWriter):

    def commit(self, **kwargs):
        super().commit(target=self._flush, **kwargs)  # reset the buffer size

    def _flush(self, call_back: Optional[Callable] = None):
        Path(self._data_path).mkdir(exist_ok=True)
        logger.info(f"Data path created: {self._data_path}")
        self._working_thread[threading.current_thread().name] = threading.current_thread()
        target_path = self._data_path
        if self._flush_count:
            target_path = Path.joinpath(self._data_path, str(self._flush_count))

        self._flush_count = self._flush_count + 1

        old_buffer = super()._new_buffer()
        file_list = old_buffer.persist(str(target_path))
        self._output_files.extend(file_list)
        if call_back:
            call_back(file_list)
        del self._working_thread[threading.current_thread().name]