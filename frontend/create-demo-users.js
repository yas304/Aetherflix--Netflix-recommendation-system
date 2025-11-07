/**
 * Script to automatically create demo users in Supabase
 * Run with: node create-demo-users.js
 */

import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://ttielvxvpabbyjkdpdlk.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0aWVsdnh2cGFiYnlqa2RwZGxrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1OTkyODcsImV4cCI6MjA3NjE3NTI4N30.t1BUevwtVv2mducL_qwrikkrM02guhMHAYjCMrMylEQ'

const supabase = createClient(supabaseUrl, supabaseAnonKey)

const demoUsers = [{
        email: 'demo1@aetherflix.com',
        password: 'demo123',
        name: 'Demo User 1'
    },
    {
        email: 'demo2@aetherflix.com',
        password: 'demo123',
        name: 'Demo User 2'
    }
]

async function createDemoUsers() {
    console.log('🚀 Creating demo users in Supabase...\n')

    for (const user of demoUsers) {
        try {
            console.log(`Creating: ${user.name} (${user.email})`)

            const { data, error } = await supabase.auth.signUp({
                email: user.email,
                password: user.password,
                options: {
                    data: {
                        name: user.name
                    }
                }
            })

            if (error) {
                if (error.message.includes('already registered')) {
                    console.log(`✅ ${user.name} already exists`)
                } else {
                    console.error(`❌ Error: ${error.message}`)
                }
            } else {
                console.log(`✅ ${user.name} created successfully!`)
            }
        } catch (err) {
            console.error(`❌ Failed to create ${user.name}:`, err.message)
        }
        console.log('')
    }

    console.log('✨ Demo user setup complete!\n')
    console.log('📋 Demo User Credentials:')
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    demoUsers.forEach((user, index) => {
        console.log(`\nUser ${index + 1}:`)
        console.log(`  Email:    ${user.email}`)
        console.log(`  Password: ${user.password}`)
        console.log(`  Name:     ${user.name}`)
    })
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    console.log('\n🎬 Visit http://localhost:5173/login to test!')
}

createDemoUsers().catch(console.error)