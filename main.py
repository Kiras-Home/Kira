"""
Kira Home Assistant - Main Entry Point
Modular, clean architecture with separated concerns
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.system_initializer import initialize_kira_system
from core.app_factory import create_kira_app
from core.configuration import setup_logging

# Setup logging
logger = setup_logging()


def main():
    """Main entry point - Keep it simple and clean"""
    print("🚀 KIRA COMPLETE SYSTEM STARTUP")
    print("=" * 60)

    try:
        # 1. Initialize all Kira systems
        print("\n📋 Initializing Kira Systems...")
        init_result = initialize_kira_system()
        
        if not init_result['success']:
            print(f"❌ System initialization failed!")
            return 1

        # 2. Create Flask application
        print("\n🌐 Creating Kira Application...")
        app = create_kira_app(init_result)
        
        # 3. Display system status
        display_startup_info(init_result)
        
        # 4. Start the application
        print(f"\n🚀 Starting Kira on http://localhost:5001")
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=True,
            use_reloader=False
        )

    except KeyboardInterrupt:
        print(f"\n👋 Kira shutdown requested by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def display_startup_info(init_result):
    """Display startup information and system status"""
    print(f"\n🌐 KIRA SYSTEM READY!")
    print(f"   🎯 Kira Ready: {init_result['kira_ready']}")
    print(f"   ✨ Full AI Experience: {init_result['full_ai_experience']}")
    print(f"   📊 Systems Available: {init_result['available_systems']}/5")

    print(f"\n🌐 ACCESS KIRA:")
    print(f"   Main Dashboard:     http://localhost:5001/")
    print(f"   Chat Interface:     http://localhost:5001/chat")
    print(f"   Memory System:      http://localhost:5001/memory")
    print(f"   System Monitor:     http://localhost:5001/system")

    print(f"\n🤖 KIRA API ENDPOINTS:")
    print(f"   Chat with Kira:     POST http://localhost:5001/api/chat/kira")
    print(f"   System Status:      GET  http://localhost:5001/api/system/status")
    print(f"   Brain Activity:     GET  http://localhost:5001/api/kira/brain-activity")

    if init_result['full_ai_experience']:
        print(f"\n🎉 FULL KIRA AI EXPERIENCE AVAILABLE!")
        print(f"   💬 Intelligent conversations with LM Studio")
        print(f"   🎤 Natural voice responses with emotions")
        print(f"   🧠 Personality and memory retention")
        print(f"   💾 Complete data storage and backup")
    elif init_result['kira_ready']:
        print(f"\n⚠️  PARTIAL KIRA EXPERIENCE")
        print(f"   Some systems may be offline - check system status")
    else:
        print(f"\n❌ KIRA SYSTEM NEEDS ATTENTION")
        print(f"   Please check LM Studio connection and system components")


if __name__ == "__main__":
    exit(main())