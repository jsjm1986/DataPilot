# -*- coding: utf-8 -*-
"""
Build Vector Index Script
Initialize schema and value indexes
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


async def main():
    print('Building vector indexes...')
    print('=' * 50)
    
    # Import here to avoid circular imports
    from src.datapilot.vector.schema_index import get_schema_index
    from src.datapilot.vector.value_index import get_value_index
    
    # Build schema index
    print('[1/2] Building schema index...')
    schema_index = get_schema_index('default')
    schema_count = await schema_index.build_index()
    print(f'  Indexed {schema_count} schema items (tables + columns)')

    # Build value index
    print('[2/2] Building value index...')
    value_index = get_value_index('default')
    value_count = await value_index.build_index()
    print(f'  Indexed {value_count} values')

    print('=' * 50)
    print(f'Total indexed: {schema_count + value_count} items')
    print('Done!')


if __name__ == '__main__':
    asyncio.run(main())
