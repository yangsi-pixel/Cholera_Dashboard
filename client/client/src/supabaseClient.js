import { createClient } from '@supabase/supabase-js'

const supabaseUrl = "https://wvykyedcdloopzyfhlbe.supabase.co"
const supabaseKey = "sb_publishable_AcbXoyx9KzVP4TOd06oMeA_oq5YYp-s"
export const supabase = createClient(supabaseUrl, supabaseKey)
