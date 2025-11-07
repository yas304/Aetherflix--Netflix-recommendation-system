import { create } from 'zustand'
import { supabase } from '../lib/supabase'

export const useAuthStore = create((set) => ({
    session: null,
    user: null,
    loading: true,

    setSession: (session) => set({ session, user: session ?.user || null }),

    initialize: async() => {
        try {
            // Set timeout to prevent hanging
            const timeoutPromise = new Promise((resolve) => {
                setTimeout(() => resolve({ data: { session: null } }), 2000)
            })

            const sessionPromise = supabase.auth.getSession()

            // Race between actual auth check and timeout
            const result = await Promise.race([sessionPromise, timeoutPromise])
            const session = result ?.data ?.session || null

            set({ session, user: session ?.user || null, loading: false })

            // Listen for auth changes
            supabase.auth.onAuthStateChange((_event, session) => {
                set({ session, user: session ?.user || null })
            })
        } catch (error) {
            console.error('Auth initialization error:', error)
            set({ session: null, user: null, loading: false })
        }
    },

    signIn: async(email, password) => {
        const { data, error } = await supabase.auth.signInWithPassword({
            email,
            password,
        })
        if (error) throw error
        set({ session: data.session, user: data.user })
        return data
    },

    signUp: async(email, password, metadata = {}) => {
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: {
                data: metadata,
            },
        })
        if (error) throw error
        return data
    },

    signOut: async() => {
        const { error } = await supabase.auth.signOut()
        if (error) throw error
        set({ session: null, user: null })
    },
}))