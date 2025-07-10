"""
To-Do List Storage Module for AI Assistant
Implements to-do list functionality using PostgreSQL storage backend
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

# Import PostgreSQL storage
from .postgresql_storage import PostgreSQLMemoryStorage

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status for to-do tasks"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    MEDIUM = 3
    HIGH = 5
    URGENT = 8
    CRITICAL = 10

class TodoStorage:
    """To-Do list storage implementation using PostgreSQL backend"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the TodoStorage with PostgreSQL backend"""
        self.storage = PostgreSQLMemoryStorage(connection_string)
        self.storage.initialize()
        self.user_id = "default"  # Default user ID
        
        logger.info("✅ TodoStorage initialized")
    
    def set_user_id(self, user_id: str):
        """Set the current user ID"""
        self.user_id = user_id
    
    def create_project(self, project_name: str, description: str = "",
                      priority: Union[TaskPriority, int] = TaskPriority.MEDIUM,
                      tags: List[str] = None) -> int:
        """
        Create a new project
        
        Args:
            project_name: Name of the project
            description: Description/idea
            priority: Priority (can be TaskPriority enum or int value)
            tags: Tags for categorization
            
        Returns:
            ID of the created project
        """
        try:
            # Convert int priority to TaskPriority if needed
            if isinstance(priority, int):
                priority_value = priority
                for p in TaskPriority:
                    if p.value == priority:
                        priority = p
                        break
            else:
                priority_value = priority.value
            
            # Create project data
            project_data = {
                "title": f"PROJECT: {project_name}",
                "description": description,
                "project_name": project_name,
                "status": TaskStatus.OPEN.value,
                "priority": priority.name,
                "priority_value": priority_value,
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Store as memory with special type
            memory_id = self.storage.store_memory(
                session_id="todo_projects",
                user_id=self.user_id,
                memory_type="todo_project",
                content=json.dumps(project_data),
                importance=priority_value,
                metadata={
                    "project_name": project_name,
                    "task_type": "project",
                    "status": TaskStatus.OPEN.value,
                    "priority": priority_value,
                    "tags": tags or []
                }
            )
            
            logger.info(f"✅ Project '{project_name}' created (ID: {memory_id})")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Error creating project: {e}")
            return None
    
    def add_task(self, project_name: str, task_title: str, description: str = "",
                priority: Union[TaskPriority, int] = TaskPriority.MEDIUM,
                due_date: Optional[str] = None, tags: List[str] = None) -> int:
        """
        Add a task to an existing project
        
        Args:
            project_name: Name of the project
            task_title: Title of the task
            description: Description of the task
            priority: Priority
            due_date: Due date (ISO format)
            tags: Tags
            
        Returns:
            ID of the created task
        """
        try:
            # Convert int priority to TaskPriority if needed
            if isinstance(priority, int):
                priority_value = priority
                for p in TaskPriority:
                    if p.value == priority:
                        priority = p
                        break
            else:
                priority_value = priority.value
            
            # Create task data
            task_data = {
                "title": task_title,
                "description": description,
                "project_name": project_name,
                "status": TaskStatus.OPEN.value,
                "priority": priority.name,
                "priority_value": priority_value,
                "due_date": due_date,
                "tags": tags or [],
                "notes": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Store as memory
            memory_id = self.storage.store_memory(
                session_id="todo_tasks",
                user_id=self.user_id,
                memory_type="todo_task",
                content=json.dumps(task_data),
                importance=priority_value,
                metadata={
                    "project_name": project_name,
                    "task_type": "task",
                    "status": TaskStatus.OPEN.value,
                    "priority": priority_value,
                    "due_date": due_date,
                    "tags": tags or []
                }
            )
            
            logger.info(f"✅ Task '{task_title}' added to project '{project_name}' (ID: {memory_id})")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Error adding task: {e}")
            return None
    
    def add_note_to_task(self, task_id: int, note: str) -> bool:
        """
        Add a note to an existing task
        
        Args:
            task_id: ID of the task
            note: Note content
            
        Returns:
            Success status
        """
        try:
            # Get the task
            task = self.get_task(task_id)
            if not task:
                logger.warning(f"⚠️ Task {task_id} not found")
                return False
            
            # Parse task data
            task_data = json.loads(task.get('content', '{}'))
            
            # Add note with timestamp
            if 'notes' not in task_data:
                task_data['notes'] = []
            
            task_data['notes'].append({
                "content": note,
                "timestamp": datetime.now().isoformat()
            })
            
            task_data['updated_at'] = datetime.now().isoformat()
            
            # Update the task
            return self.update_task(task_id, task_data)
            
        except Exception as e:
            logger.error(f"❌ Error adding note: {e}")
            return False
    
    def update_task_status(self, task_id: int, status: Union[TaskStatus, str]) -> bool:
        """
        Update the status of a task
        
        Args:
            task_id: ID of the task
            status: New status (TaskStatus enum or string)
            
        Returns:
            Success status
        """
        try:
            # Convert string status to TaskStatus if needed
            if isinstance(status, str):
                status_value = status
                for s in TaskStatus:
                    if s.value == status:
                        status = s
                        break
            else:
                status_value = status.value
            
            # Get the task
            task = self.get_task(task_id)
            if not task:
                logger.warning(f"⚠️ Task {task_id} not found")
                return False
            
            # Parse task data
            task_data = json.loads(task.get('content', '{}'))
            
            # Update status
            task_data['status'] = status_value
            task_data['updated_at'] = datetime.now().isoformat()
            
            # Update the task
            return self.update_task(task_id, task_data)
            
        except Exception as e:
            logger.error(f"❌ Error updating task status: {e}")
            return False
    
    def update_task(self, task_id: int, task_data: Dict) -> bool:
        """
        Update a task with new data
        
        Args:
            task_id: ID of the task
            task_data: New task data
            
        Returns:
            Success status
        """
        try:
            # Get the original task to preserve ID
            original_task = self.get_task(task_id)
            if not original_task:
                logger.warning(f"⚠️ Task {task_id} not found")
                return False
            
            # Update metadata
            metadata = {
                "project_name": task_data.get('project_name', ''),
                "task_type": "task",
                "status": task_data.get('status', TaskStatus.OPEN.value),
                "priority": task_data.get('priority_value', TaskPriority.MEDIUM.value),
                "due_date": task_data.get('due_date'),
                "tags": task_data.get('tags', [])
            }
            
            # Store updated task
            with self.storage.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Update the memory_entries record
                    update_sql = '''
                        UPDATE memory_entries
                        SET content = %s,
                            metadata = %s,
                            importance = %s,
                            last_accessed = NOW(),
                            access_count = access_count + 1
                        WHERE id = %s
                    '''
                    
                    cursor.execute(
                        update_sql,
                        (
                            json.dumps(task_data),
                            json.dumps(metadata),
                            task_data.get('priority_value', TaskPriority.MEDIUM.value),
                            task_id
                        )
                    )
                    
                    success = cursor.rowcount > 0
                    if success:
                        logger.info(f"✅ Task {task_id} updated successfully")
                    else:
                        logger.warning(f"⚠️ Task {task_id} update failed - no rows affected")
                    
                    return success
                    
                finally:
                    cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Error updating task: {e}")
            return False
    
    def get_task(self, task_id: int) -> Optional[Dict]:
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task data or None if not found
        """
        try:
            with self.storage.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=self.storage._get_dict_cursor())
                try:
                    cursor.execute(
                        "SELECT * FROM memory_entries WHERE id = %s",
                        (task_id,)
                    )
                    
                    task = cursor.fetchone()
                    if task:
                        # Update access timestamp
                        self.storage.update_memory_access(task_id)
                        return dict(task)
                    
                    return None
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Error getting task: {e}")
            return None
    
    def get_projects(self) -> List[Dict]:
        """
        Get all projects for the current user
        
        Returns:
            List of projects
        """
        try:
            return self.storage.search_memories(
                user_id=self.user_id,
                memory_type="todo_project",
                session_id="todo_projects",
                limit=100
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting projects: {e}")
            return []
    
    def get_tasks_by_project(self, project_name: str) -> List[Dict]:
        """
        Get all tasks for a specific project
        
        Args:
            project_name: Name of the project
            
        Returns:
            List of tasks
        """
        try:
            with self.storage.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=self.storage._get_dict_cursor())
                try:
                    cursor.execute(
                        """
                        SELECT * FROM memory_entries 
                        WHERE user_id = %s 
                        AND memory_type = 'todo_task' 
                        AND metadata->>'project_name' = %s
                        ORDER BY importance DESC, created_at DESC
                        """,
                        (self.user_id, project_name)
                    )
                    
                    tasks = cursor.fetchall()
                    return [dict(task) for task in tasks]
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Error getting tasks by project: {e}")
            return []
    
    def search_tasks(self, query: str, status: Optional[str] = None) -> List[Dict]:
        """
        Search tasks by query string and optional status
        
        Args:
            query: Search query
            status: Optional status filter
            
        Returns:
            List of matching tasks
        """
        try:
            with self.storage.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=self.storage._get_dict_cursor())
                try:
                    sql = """
                        SELECT * FROM memory_entries 
                        WHERE user_id = %s 
                        AND memory_type = 'todo_task' 
                        AND content ILIKE %s
                    """
                    params = [self.user_id, f"%{query}%"]
                    
                    if status:
                        sql += " AND metadata->>'status' = %s"
                        params.append(status)
                    
                    sql += " ORDER BY importance DESC, created_at DESC"
                    
                    cursor.execute(sql, params)
                    
                    tasks = cursor.fetchall()
                    return [dict(task) for task in tasks]
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Error searching tasks: {e}")
            return []
    
    def delete_task(self, task_id: int) -> bool:
        """
        Delete a task
        
        Args:
            task_id: ID of the task
            
        Returns:
            Success status
        """
        try:
            with self.storage.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "DELETE FROM memory_entries WHERE id = %s",
                        (task_id,)
                    )
                    
                    success = cursor.rowcount > 0
                    if success:
                        logger.info(f"✅ Task {task_id} deleted successfully")
                    else:
                        logger.warning(f"⚠️ Task {task_id} deletion failed - not found")
                    
                    return success
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Error deleting task: {e}")
            return False
    
    def get_task_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tasks for the current user
        
        Returns:
            Summary statistics
        """
        try:
            with self.storage.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=self.storage._get_dict_cursor())
                try:
                    # Get project count
                    cursor.execute(
                        "SELECT COUNT(*) FROM memory_entries WHERE user_id = %s AND memory_type = 'todo_project'",
                        (self.user_id,)
                    )
                    project_count = cursor.fetchone()['count']
                    
                    # Get task counts by status
                    cursor.execute(
                        """
                        SELECT metadata->>'status' as status, COUNT(*) 
                        FROM memory_entries 
                        WHERE user_id = %s AND memory_type = 'todo_task'
                        GROUP BY metadata->>'status'
                        """,
                        (self.user_id,)
                    )
                    
                    status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
                    
                    # Get high priority tasks
                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM memory_entries 
                        WHERE user_id = %s 
                        AND memory_type = 'todo_task' 
                        AND importance >= 5
                        """,
                        (self.user_id,)
                    )
                    high_priority_count = cursor.fetchone()['count']
                    
                    return {
                        "total_projects": project_count,
                        "task_counts": status_counts,
                        "total_tasks": sum(status_counts.values()),
                        "high_priority_tasks": high_priority_count,
                        "open_tasks": status_counts.get(TaskStatus.OPEN.value, 0),
                        "completed_tasks": status_counts.get(TaskStatus.COMPLETED.value, 0)
                    }
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Error getting task summary: {e}")
            return {
                "total_projects": 0,
                "task_counts": {},
                "total_tasks": 0,
                "high_priority_tasks": 0,
                "open_tasks": 0,
                "completed_tasks": 0,
                "error": str(e)
            }