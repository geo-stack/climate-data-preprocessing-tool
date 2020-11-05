# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © SARDES Project Contributors
# https://github.com/cgq-qgc/sardes
#
# This file is part of SARDES.
# Licensed under the terms of the MIT License.
# -----------------------------------------------------------------------------

# ---- Standard imports
from collections import OrderedDict
from time import sleep
import uuid

# ---- Third party imports
from qtpy.QtCore import QObject, QThread, Signal, Slot


class WorkerBase(QObject):
    """
    A worker to execute tasks without blocking the gui.
    """
    sig_task_completed = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.project_accessor = None
        self._tasks = OrderedDict()

    def add_task(self, task_uuid4, task, *args, **kargs):
        """
        Add a task to the stack that will be executed when the thread of
        this worker is started.
        """
        self._tasks[task_uuid4] = (task, args, kargs)

    def run_tasks(self):
        """Execute the tasks that were added to the stack."""
        for task_uuid4, (task, args, kargs) in self._tasks.items():
            method_to_exec = getattr(self, task)
            returned_values = method_to_exec(*args, **kargs)
            self.sig_task_completed.emit(task_uuid4, returned_values)
        self._tasks = OrderedDict()
        self.thread().quit()


class TaskManagerBase(QObject):
    """
    A basic manager to handle tasks that need to be executed in a different
    thread than that of the main application to avoid blocking the GUI event
    loop.
    """
    sig_run_tasks_finished = Signal()

    def __init__(self):
        super().__init__()
        self._worker = None

        self._task_callbacks = {}
        self._task_data = {}

        self._running_tasks = []
        self._queued_tasks = []
        self._pending_tasks = []
        # Queued tasks are tasks whose execution has not been requested yet.
        # This happens when we want the Worker to execute a list of tasks
        # in a single run. All queued tasks are dumped in the list of pending
        # tasks when `run_task` is called.
        #
        # Pending tasks are tasks whose execution was postponed due to
        # the fact that the worker was busy. These tasks are run as soon
        # as the worker becomes available.
        #
        # Running tasks are tasks that are being executed by the worker.

    def run_tasks(self):
        """
        Execute all the tasks that were added to the stack.
        """
        self._run_tasks()

    def add_task(self, task, callback, *args, **kargs):
        self._add_task(task, callback, *args, **kargs)

    def worker(self):
        """Return the worker that is installed on this manager."""
        return self._worker

    def set_worker(self, worker):
        """"Install the provided worker on this manager"""
        self._worker = worker
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run_tasks)

        # Connect the worker signals to handlers.
        self._worker.sig_task_completed.connect(
            self._exec_task_callback)

    # ---- Private API
    @Slot(object, object)
    def _exec_task_callback(self, task_uuid4, returned_values):
        """
        This is the (only) slot that is called after a task is completed
        by the worker.
        """
        # Run the callback associated with the specified task UUID if any.
        if self._task_callbacks[task_uuid4] is not None:
            try:
                self._task_callbacks[task_uuid4](*returned_values)
            except TypeError:
                self._task_callbacks[task_uuid4]()

        # Clean up internal variables.
        del self._task_callbacks[task_uuid4]
        del self._task_data[task_uuid4]
        self._running_tasks.remove(task_uuid4)

        if len(self._running_tasks) == 0:
            # This means all tasks sent to the worker were completed.
            if len(self._pending_tasks) > 0:
                self._run_pending_tasks()
            else:
                self.sig_run_tasks_finished.emit()
                print('All pending tasks were executed.')

    def _add_task(self, task, callback, *args, **kargs):
        task_uuid4 = uuid.uuid4()
        self._task_callbacks[task_uuid4] = callback
        self._queued_tasks.append(task_uuid4)
        self._task_data[task_uuid4] = (task, args, kargs)

    def _run_tasks(self):
        """
        Execute all the tasks that were added to the stack.
        """
        self._pending_tasks.extend(self._queued_tasks)
        self._queued_tasks = []
        self._run_pending_tasks()

    def _run_pending_tasks(self):
        """Execute all pending tasks."""
        if len(self._running_tasks) == 0:
            print('Executing {} pending tasks...'.format(
                len(self._pending_tasks)))
            # Even though the worker has executed all its tasks,
            # we may still need to wait a little for it to stop properly.
            i = 0
            while self._thread.isRunning():
                sleep(0.1)
                i += 1
                if i > 100:
                    print("Error: unable to stop {}'s working thread.".format(
                        self.__class__.__name__))

            self._running_tasks = self._pending_tasks.copy()
            self._pending_tasks = []
            for task_uuid4 in self._running_tasks:
                task, args, kargs = self._task_data[task_uuid4]
                self._worker.add_task(task_uuid4, task, *args, **kargs)
            self._thread.start()
