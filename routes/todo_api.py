"""
To-Do List API Routes
Provides API endpoints for managing to-do lists
"""

import json
import logging
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify

# Import the TodoStorage class
from memory.storage.todo_storage import TodoStorage, TaskStatus, TaskPriority

# Create logger
logger = logging.getLogger(__name__)

# Create Blueprint
todo_api = Blueprint('todo_api', __name__)

# Initialize TodoStorage
todo_storage = TodoStorage()

@todo_api.route('/api/todo/projects', methods=['GET'])
def get_projects():
    """Get all projects for the current user"""
    try:
        # Get user_id from request or use default
        user_id = request.args.get('user_id', 'default')
        todo_storage.set_user_id(user_id)
        
        projects = todo_storage.get_projects()
        
        # Process projects to extract data from content
        processed_projects = []
        for project in projects:
            try:
                project_data = json.loads(project.get('content', '{}'))
                processed_projects.append({
                    'id': project.get('id'),
                    'project_name': project_data.get('project_name'),
                    'title': project_data.get('title'),
                    'description': project_data.get('description'),
                    'status': project_data.get('status'),
                    'priority': project_data.get('priority'),
                    'priority_value': project_data.get('priority_value'),
                    'tags': project_data.get('tags', []),
                    'created_at': project_data.get('created_at'),
                    'updated_at': project_data.get('updated_at')
                })
            except Exception as e:
                logger.error(f"Error processing project: {e}")
        
        return jsonify({
            'status': 'success',
            'projects': processed_projects
        })
    
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Get required fields
        project_name = data.get('project_name')
        if not project_name:
            return jsonify({
                'status': 'error',
                'message': 'Project name is required'
            }), 400
        
        # Get optional fields
        description = data.get('description', '')
        priority = data.get('priority', 3)  # Default to MEDIUM
        tags = data.get('tags', [])
        user_id = data.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Create project
        project_id = todo_storage.create_project(
            project_name=project_name,
            description=description,
            priority=priority,
            tags=tags
        )
        
        if project_id:
            return jsonify({
                'status': 'success',
                'project_id': project_id,
                'message': f'Project {project_name} created successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create project'
            }), 500
    
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/tasks', methods=['POST'])
def create_task():
    """Create a new task"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Get required fields
        project_name = data.get('project_name')
        task_title = data.get('title')
        
        if not project_name or not task_title:
            return jsonify({
                'status': 'error',
                'message': 'Project name and task title are required'
            }), 400
        
        # Get optional fields
        description = data.get('description', '')
        priority = data.get('priority', 3)  # Default to MEDIUM
        due_date = data.get('due_date')
        tags = data.get('tags', [])
        user_id = data.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Create task
        task_id = todo_storage.add_task(
            project_name=project_name,
            task_title=task_title,
            description=description,
            priority=priority,
            due_date=due_date,
            tags=tags
        )
        
        if task_id:
            return jsonify({
                'status': 'success',
                'task_id': task_id,
                'message': f'Task {task_title} created successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create task'
            }), 500
    
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/tasks/<int:task_id>/notes', methods=['POST'])
def add_note_to_task(task_id):
    """Add a note to a task"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Get required fields
        note = data.get('note')
        if not note:
            return jsonify({
                'status': 'error',
                'message': 'Note content is required'
            }), 400
        
        # Get optional fields
        user_id = data.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Add note
        success = todo_storage.add_note_to_task(task_id, note)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Note added to task {task_id} successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to add note to task {task_id}'
            }), 500
    
    except Exception as e:
        logger.error(f"Error adding note to task: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/tasks/<int:task_id>/status', methods=['PUT'])
def update_task_status(task_id):
    """Update task status"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Get required fields
        new_status = data.get('status')
        if not new_status:
            return jsonify({
                'status': 'error',
                'message': 'Status is required'
            }), 400
        
        # Get optional fields
        user_id = data.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Update status
        success = todo_storage.update_task_status(task_id, new_status)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Task {task_id} status updated to {new_status} successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to update task {task_id} status'
            }), 500
    
    except Exception as e:
        logger.error(f"Error updating task status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a task"""
    try:
        # Get optional fields
        user_id = request.args.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Delete task
        success = todo_storage.delete_task(task_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Task {task_id} deleted successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to delete task {task_id}'
            }), 500
    
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/projects/<project_name>/tasks', methods=['GET'])
def get_tasks_by_project(project_name):
    """Get all tasks for a project"""
    try:
        # Get optional fields
        user_id = request.args.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Get tasks
        tasks = todo_storage.get_tasks_by_project(project_name)
        
        # Process tasks to extract data from content
        processed_tasks = []
        for task in tasks:
            try:
                task_data = json.loads(task.get('content', '{}'))
                processed_tasks.append({
                    'id': task.get('id'),
                    'project_name': task_data.get('project_name'),
                    'title': task_data.get('title'),
                    'description': task_data.get('description'),
                    'status': task_data.get('status'),
                    'priority': task_data.get('priority'),
                    'priority_value': task_data.get('priority_value'),
                    'due_date': task_data.get('due_date'),
                    'tags': task_data.get('tags', []),
                    'notes': task_data.get('notes', []),
                    'created_at': task_data.get('created_at'),
                    'updated_at': task_data.get('updated_at')
                })
            except Exception as e:
                logger.error(f"Error processing task: {e}")
        
        return jsonify({
            'status': 'success',
            'tasks': processed_tasks
        })
    
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/search', methods=['GET'])
def search_tasks():
    """Search tasks"""
    try:
        # Get query parameters
        query = request.args.get('q', '')
        status = request.args.get('status')
        user_id = request.args.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Search tasks
        tasks = todo_storage.search_tasks(query, status)
        
        # Process tasks to extract data from content
        processed_tasks = []
        for task in tasks:
            try:
                task_data = json.loads(task.get('content', '{}'))
                processed_tasks.append({
                    'id': task.get('id'),
                    'project_name': task_data.get('project_name'),
                    'title': task_data.get('title'),
                    'description': task_data.get('description'),
                    'status': task_data.get('status'),
                    'priority': task_data.get('priority'),
                    'priority_value': task_data.get('priority_value'),
                    'due_date': task_data.get('due_date'),
                    'tags': task_data.get('tags', []),
                    'notes': task_data.get('notes', []),
                    'created_at': task_data.get('created_at'),
                    'updated_at': task_data.get('updated_at')
                })
            except Exception as e:
                logger.error(f"Error processing task: {e}")
        
        return jsonify({
            'status': 'success',
            'tasks': processed_tasks
        })
    
    except Exception as e:
        logger.error(f"Error searching tasks: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@todo_api.route('/api/todo/summary', methods=['GET'])
def get_task_summary():
    """Get task summary"""
    try:
        # Get optional fields
        user_id = request.args.get('user_id', 'default')
        
        # Set user ID
        todo_storage.set_user_id(user_id)
        
        # Get summary
        summary = todo_storage.get_task_summary()
        
        return jsonify({
            'status': 'success',
            'summary': summary
        })
    
    except Exception as e:
        logger.error(f"Error getting task summary: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500